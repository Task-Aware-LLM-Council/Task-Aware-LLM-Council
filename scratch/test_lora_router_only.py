"""Test the LoRA adapter's routing output directly — no specialists, no synth.

Loads the joint decomposer+router with the LoRA adapter, feeds a handful of
prompts covering all three specialist roles, prints the decomposition. Goal:
confirm the adapter learned non-trivial routing (not just `math_code` for
everything like zero-shot).

Run from repo root on a CARC compute node:

    uv run --package council-policies python scratch/test_lora_router_only.py

Takes ~90s (model load) + ~5s/prompt on CPU, faster on GPU.
"""
import asyncio

from council_policies import HFCausalGenerate, Seq2SeqDecomposerRouter

ADAPTER = "artifacts/decomposer_router_causal/adapter"

# TODO(you): pick prompts that clearly isolate each role so failures are obvious.
# Edit these to match your dataset's phrasing — the point is to see whether the
# LoRA routes single-skill prompts to ONE role and multi-skill prompts to
# multiple roles in the right order.
PROMPTS = [
    # Expect: math_code, single subtask
    "Compute the definite integral of x^3 from 0 to 2.",
    # Expect: fact_general, single subtask
    "What is the capital of Australia?",
    # Expect: qa_reasoning, single subtask
    "Explain why the sky appears blue during the day but red at sunset.",
    # Expect: 2 subtasks, math_code then qa_reasoning
    "Compute 5 factorial, then explain whether the result is a perfect square.",
    # Expect: fact_general then qa_reasoning
    "Who won the 2018 World Cup, and why was that result historically significant?",
]


async def main() -> None:
    print(f"Loading base + LoRA from {ADAPTER} ...")
    # Change device="cpu" to "cuda" if you have a GPU free.
    generate_fn = HFCausalGenerate(
        "google/gemma-2-2b-it", peft_adapter=ADAPTER, device="cpu",
    )
    decomposer = Seq2SeqDecomposerRouter(generate_fn=generate_fn, max_subtasks=4)
    print("Ready. Routing prompts:\n")

    for i, prompt in enumerate(PROMPTS):
        print(f"--- prompt {i} ---")
        print(f"Q: {prompt}")
        targets = await decomposer.decompose(prompt, context="")
        for t in targets:
            print(f"  -> [{t.suggested_role}] {t.text}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
