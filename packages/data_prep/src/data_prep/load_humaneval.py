import random
from datasets import load_dataset

from common.ids import make_id
from common.schema import RouterExample
from common.utils import to_text


def load_humaneval(n: int = 40, seed: int = 42) -> list[dict]:
    print(f"Loading HumanEval+ ({n} samples)...")
    ds = load_dataset(
        "openai/openai_humaneval",
        split="test",
        streaming=True
    )

    ds = ds.shuffle(seed=seed, buffer_size=1000)

    sampled = []
    for ex in ds:
        sampled.append(ex)
        if len(sampled) >= n:
            break

    records = []
    for ex in sampled:
        task_id = to_text(ex.get("task_id", ""))

        rec = RouterExample(
            id=make_id("HumanEvalPlus", task_id),
            source_dataset="HumanEvalPlus",
            question=to_text(ex.get("prompt", "")),
            context="",
            gold_answer=to_text(ex.get("canonical_solution", "")),
            gold_label="",
            unit_tests=to_text(ex.get("test", "")),
            skill_tags=["code"],
            hallucination_subset=False,
            split="",
            original_id=task_id,
            metadata={
                "entry_point" : str(ex.get("entry_point", ""))
            }
        )
        records.append(rec.to_dict())

    return records
