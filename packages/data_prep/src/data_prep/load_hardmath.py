import random
from datasets import load_dataset

from common.ids import make_id
from common.schema import RouterExample
from common.utils import to_text

HARDER_PROBLEMS_LEVEL = (4, 5)


def load_hardmath(n: int = 40, seed: int = 42) -> list[dict]:
    print(f"Loading HARDMath ({n} samples)...")
    ds = load_dataset(
        "math-ai/MATH-500",
        split="test",
        streaming=True
    )

    ds = ds.shuffle(seed=seed, buffer_size=1000)

    sampled = []
    for ex in ds:
        # Filter for harder problems (level 4-5)
        if ex.get("level", 0) in HARDER_PROBLEMS_LEVEL:
            sampled.append(ex)

        if len(sampled) >= n:
            break

    records = []
    for ex in sampled:
        orig_id = ex.get(
            "unique_id", f"{ex.get('subject', '')}_{ex.get('level', '')}"
        )
        rec = RouterExample(
            id=make_id("HARDMATH", ex.get(
                "unique_id", ex.get("problem", "")[:50])),
            source_dataset="HARDMATH",
            question=to_text(ex.get("problem", "")),
            context="",
            gold_answer=to_text(ex.get("solution", "")),
            gold_label="",
            unit_tests="",
            skill_tags=["math"],
            hallucination_subset=False,
            split="",
            original_id=to_text(orig_id),
        )
        records.append(rec.to_dict())

    return records
