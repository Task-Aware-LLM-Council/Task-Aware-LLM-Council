"""
Joint decomposer + router — seq2seq serving class.

One model replaces `LLMDecomposer` + `LearnedRouter`. A single
`generate_fn` call produces a JSON array of
`{"role": <math_code|qa_reasoning|fact_general>, "subtask": str}`
pairs; the routing label is baked into the decomposer output, so the
policy no longer has to consult a separate router per subtask.

Adapter seam
------------
`GenerateFn` is text-in / text-out. The production adapter
(`HFSeq2SeqGenerate`) wraps a Flan-T5-small fine-tune loaded via HF
`transformers`, `torch.inference_mode()`, single load per process.
Unit tests inject a canned-output fake so the parsing/fallback logic
is exercised without torch.

Why the shared input format
---------------------------
The training script (`training.train_decomposer_router`) and this
module both call `featurize(...)` with the same kwargs, then prepend
the same `INPUT_PREFIX`. Any drift between them silently degrades
accuracy — the `input_format_version` field on `DecomposerRouterCard`
is the tripwire.

Why the parse helper is shared
------------------------------
`_parse_subtask_array` in `decomposer.py` handles the JSON-array
ladder for both `LLMDecomposer` (external LLM call) and this class
(seq2seq model). Same fallback rules → same behavior on malformed
output, regardless of which decomposer is wired in.

Role vocab is NOT validated here
--------------------------------
The class emits `Subtask.suggested_role` verbatim (stripped). Role
validation — "is this role in ROLE_LABELS and registered on the
orchestrator?" — happens in `LearnedRouterPolicy.plan()` when
`use_joint_model=True`. Keeping validation in the policy lets it
reach `PolicyRuntime.has_specialist_role()`, which this module does
not have a handle to.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from council_policies.decomposer import _parse_subtask_array
from council_policies.router import Subtask
from council_policies.router_featurize import DEFAULT_CONTEXT_CHAR_CAP, featurize
from council_policies.router_labels import DEFAULT_FALLBACK_ROLE

logger = logging.getLogger(__name__)

INPUT_PREFIX: str = "decompose_and_route: "
"""Task-prefix that the fine-tuned model is conditioned on at training
time. Serving must use the exact same prefix — typo here silently
wrecks accuracy. Also imported by `training.train_decomposer_router`
so both sides share one string."""


# --------------------------------------------------------------------------- #
# Adapter seam
# --------------------------------------------------------------------------- #


class GenerateFn(Protocol):
    """Text in → text out. Mirrors `ScoreFn` in `router.py`: decouples
    the serving-class logic from torch/transformers so unit tests can
    inject canned responses."""

    def __call__(self, text: str) -> str: ...


# --------------------------------------------------------------------------- #
# Serving class
# --------------------------------------------------------------------------- #


class Seq2SeqDecomposerRouter:
    """
    Satisfies the `Decomposer` protocol. One `generate_fn` call per
    prompt; emits `Subtask`s whose `suggested_role` is the joint
    model's routing decision.

    Parameters
    ----------
    generate_fn:
        Any `GenerateFn`. Production wires `HFSeq2SeqGenerate`; tests
        pass a fake.
    max_subtasks:
        Hard ceiling enforced at parse time. Even if the model emits
        more, the parser truncates — defense in depth against a
        runaway generation.
    context_char_cap:
        Upstream char cap on context before `featurize` tokenizes —
        pinned to the same default the training script used. Mismatch
        is the most common source of silent accuracy drops.
    fallback_role:
        Role the `Subtask` gets tagged with if the joint model emits
        a missing/empty `role` field. Defaults to the generalist. Note
        that *unknown-role* handling (model emits a string not in
        `ROLE_LABELS`) happens in `LearnedRouterPolicy.plan()`, not
        here — see module docstring.

    Fallback behavior
    -----------------
    On any generation/parse failure (parse ladder defined in
    `_parse_subtask_array`), returns a single passthrough `Subtask`
    with `suggested_role=None` — the policy then routes through its
    own `fallback_role`.
    """

    def __init__(
        self,
        *,
        generate_fn: GenerateFn,
        max_subtasks: int = 4,
        context_char_cap: int = DEFAULT_CONTEXT_CHAR_CAP,
        fallback_role: str = DEFAULT_FALLBACK_ROLE,
    ) -> None:
        if max_subtasks < 1:
            raise ValueError(f"max_subtasks must be >= 1, got {max_subtasks}")
        if context_char_cap < 0:
            raise ValueError(
                f"context_char_cap must be >= 0, got {context_char_cap}"
            )
        self._generate_fn = generate_fn
        self.max_subtasks = max_subtasks
        self.context_char_cap = context_char_cap
        self.fallback_role = fallback_role

    async def decompose(
        self, prompt: str, context: str = "",
    ) -> list[Subtask]:
        model_input = INPUT_PREFIX + featurize(
            prompt, context, context_char_cap=self.context_char_cap,
        )
        try:
            raw = self._generate_fn(model_input)
        except Exception as exc:
            logger.warning(
                "Seq2SeqDecomposerRouter: generate_fn raised (%s); "
                "falling back to passthrough.",
                type(exc).__name__,
            )
            return [Subtask(text=prompt, order=0)]

        return _parse_subtask_array(
            raw,
            fallback_prompt=prompt,
            max_subtasks=self.max_subtasks,
        )


# --------------------------------------------------------------------------- #
# HuggingFace-backed adapter
# --------------------------------------------------------------------------- #


class HFSeq2SeqGenerate:
    """
    Production `GenerateFn`: HuggingFace `AutoModelForSeq2SeqLM` loaded
    once per process.

    Loaded in FP16 by default (Flan-T5-small → ~150MB VRAM). Generation
    config is pulled from the saved artifact's `generation_config.json`
    when present, so the num_beams / max_new_tokens from training come
    along automatically.

    Not vLLM-compatible (see module docstring); runs in-process.

    Parameters
    ----------
    model_dir:
        Path or HF hub ID of the trained artifact. Expects
        `generation_config.json` written by `train_decomposer_router`
        to sit alongside the model weights — if absent, falls back to
        HF's default generation config.
    device:
        `"auto"` → `"cuda"` if available else `"cpu"`. Explicit string
        overrides. CPU fallback adds ~100ms/call — acceptable when
        specialists already saturate the GPU.
    torch_dtype:
        Defaults to fp16 on CUDA, fp32 on CPU (fp16 CPU is slower and
        not well supported).
    max_input_length:
        Upstream truncation. Must be ≥ the `--input-max-length` used
        at training. Default 512 matches the training default.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "auto",
        torch_dtype: Any | None = None,
        max_input_length: int = 512,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            GenerationConfig,
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_dtype is None:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype,
        )
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._max_input_length = max_input_length

        try:
            self._generation_config = GenerationConfig.from_pretrained(model_dir)
        except (OSError, ValueError):
            logger.warning(
                "HFSeq2SeqGenerate: no generation_config.json at %s; "
                "using HF default.",
                model_dir,
            )
            self._generation_config = None

    def __call__(self, text: str) -> str:
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_length,
        ).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                generation_config=self._generation_config,
            )

        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
