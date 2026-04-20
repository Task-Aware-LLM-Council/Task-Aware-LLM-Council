"""Causal-LM `GenerateFn` adapter — drop-in for `HFSeq2SeqGenerate`.

Lets `Seq2SeqDecomposerRouter` wrap a decoder-only chat model (Gemma-2,
Qwen2.5, Llama-3, ...) instead of the Flan-T5 fine-tune. Useful for
zero-shot evaluation of the joint decomposer+router task before
committing to training — if a 2-3B instruct model already emits
parseable output from a well-crafted system prompt, the training budget
can be spent elsewhere.

The adapter implements `GenerateFn`: it takes the string produced by
`Seq2SeqDecomposerRouter.decompose` (which prepends the T5 task prefix
and featurizes prompt+context), embeds it as the user turn of a chat
template together with `DECOMPOSER_ROUTER_SYSTEM_PROMPT`, and returns
the newly generated tokens. The parse/fallback ladder in the serving
class works unchanged because the output format matches `parse_targets`.

Gemma-2 specifics
-----------------
Gemma's chat template has no `system` role. We prepend the system
instructions into the first user message — this is the convention in
Gemma's own documentation. The adapter uses `apply_chat_template` so
any other chat model (Qwen2.5, Llama-3) that has a `system` role can
still use this adapter once we add a branch for it, but for a
single-user-message layout works for both.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


DECOMPOSER_ROUTER_SYSTEM_PROMPT = """\
You decompose a user prompt into the minimum number of independent \
subtasks needed to answer it fully, and route each subtask to exactly \
one specialist role.

Specialist roles:
  math_code     - computation, algorithms, code generation or debugging
  qa_reasoning  - multi-hop reasoning, long-context QA, explanation
  fact_general  - fact retrieval, fact verification, general knowledge

Rules:
- If the prompt needs only one role, output a single line. Do not split \
single-skill prompts into reasoning sub-steps.
- If the prompt needs two or more distinct roles, output one line per \
subtask in execution order.
- Each subtask must be self-contained: include enough context that a \
specialist who has not seen the other subtasks can answer it.
- Output format, EXACTLY: one subtask per line as `<role>: <subtask>`. \
No preamble, no JSON, no markdown, no blank lines, no trailing commentary.

Examples:

Input: Compute 2+2 and explain whether 4 is a prime number.
Output:
math_code: Compute 2+2.
qa_reasoning: Explain whether 4 is a prime number.

Input: What is the capital of France?
Output:
fact_general: Identify the capital city of France.
"""


class HFCausalGenerate:
    """Production `GenerateFn` for instruction-tuned causal LMs.

    Loads a HuggingFace `AutoModelForCausalLM` once per process. Each
    call applies the model's chat template (system + user), generates
    greedily, and returns only the newly generated tokens so
    `parse_targets` sees raw output.

    Parameters
    ----------
    model_id:
        HF hub ID (e.g. ``google/gemma-2-2b-it``) or a local path.
    system_prompt:
        Defaults to `DECOMPOSER_ROUTER_SYSTEM_PROMPT`, which instructs
        the model to emit `role: subtask` lines matching the
        `parse_targets` contract.
    device:
        ``auto`` picks CUDA when available, else CPU. CPU inference is
        usable for a 2B model but ~5-10x slower per request.
    torch_dtype:
        Defaults to bf16 on CUDA (Ampere+), fp32 on CPU. Override to
        fp16 on pre-Ampere GPUs or to int8/int4 via bitsandbytes if
        VRAM is tight.
    max_input_length:
        Truncation cap on the chat-formatted prompt. 4096 leaves room
        for the system prompt (~350 tokens) plus most long-context
        featurized inputs; QuALITY rows get clipped.
    max_new_tokens:
        Generation cap. 256 is ample for ≤4 ``role: subtask`` lines.
    """

    def __init__(
        self,
        model_id: str,
        *,
        peft_adapter: str | None = None,
        system_prompt: str = DECOMPOSER_ROUTER_SYSTEM_PROMPT,
        device: str = "auto",
        torch_dtype: Any | None = None,
        max_input_length: int = 4096,
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Load tokenizer from the adapter if provided — it was saved alongside
        # and may carry added special tokens — else fall back to the base.
        tokenizer_src = peft_adapter or model_id
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype,
        )
        if peft_adapter:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, peft_adapter)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._system_prompt = system_prompt
        self._max_input_length = max_input_length
        self._max_new_tokens = max_new_tokens

    def __call__(self, text: str) -> str:
        import torch

        user_msg = f"{self._system_prompt}\n\nInput: {text}\nOutput:"
        chat_prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_length,
        ).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Slice off the prompt tokens so only the newly generated content
        # is returned — parse_targets must not see the echoed input.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
