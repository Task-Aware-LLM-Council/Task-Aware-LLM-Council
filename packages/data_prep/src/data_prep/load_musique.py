import random
from datasets import load_dataset

from common.ids import make_id
from common.schema import RouterExample
from common.utils import to_text

MIN_QUESTION_LENGTH = 30


def load_musique(n: int = 40, seed: int = 42) -> list[dict]:
    print(f"Loading MuSiQue ({n} samples)...")

    # Enable streaming to bypass the disk download
    ds = load_dataset(
        "bdsaglam/musique",
        split="validation",
        streaming=True
    )

    # Maintain a rolling buffer to randomize the incoming stream
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    # Stream and Early Stop
    sampled = []
    for ex in ds:
        # Filter for non-trivial questions
        if len(ex.get("question", "")) > MIN_QUESTION_LENGTH:
            sampled.append(ex)

        # Sever the network connection the moment we hit our target (n)
        if len(sampled) >= n:
            break

    # Format into Router schema
    records = []
    for ex in sampled:
        paragraphs = ex.get("paragraphs", [])

        if isinstance(paragraphs, list) and paragraphs:
            if isinstance(paragraphs[0], dict):
                context = "\n\n".join(
                    p.get("paragraph_text", p.get("text", to_text(p)))
                    for p in paragraphs
                )
            else:
                context = "\n\n".join(to_text(p) for p in paragraphs)
        else:
            context = ""

        rec = RouterExample(
            id=make_id("MuSiQue", ex.get("id", random.randint(0, 999999))),
            source_dataset="MuSiQue",
            question=to_text(ex.get("question", "")),
            context=context,
            gold_answer=to_text(ex.get("answer", "")),
            gold_label="",
            unit_tests="",
            skill_tags=["retrieval", "multi-hop"],
            hallucination_subset=False,
            split="",
            original_id=to_text(ex.get("id", "")),
        )
        records.append(rec.to_dict())

    return records
