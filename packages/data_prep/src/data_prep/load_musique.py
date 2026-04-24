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
    

    # Keep only 10% unanswerable rows
    unanswerable_cap = max(1, int(round(n * 0.1)))
    answerable_cap = n - unanswerable_cap

    answerable_count = 0
    unanswerable_count = 0
    sampled = []

    for ex in ds:
        question = ex.get("question", "")
        if len(question) <= MIN_QUESTION_LENGTH:
            continue

        is_answerable = bool(ex.get("answerable", False))

        if is_answerable and answerable_count < answerable_cap:
            sampled.append(ex)
            answerable_count += 1
        elif not is_answerable and unanswerable_count < unanswerable_cap:
            sampled.append(ex)
            unanswerable_count += 1

        if len(sampled) >= n:
            break

    # Format into Router schema
    records = []
    for ex in sampled:
        context = ex.get("paragraphs", dict())
        # if isinstance(paragraphs, list) and paragraphs:
        #     if isinstance(paragraphs[0], dict):
        #         context = "\n\n".join(
        #             p.get("paragraph_text", p.get("text", to_text(p)))
        #             for p in paragraphs
        #         )
        #     else:
        #         context = "\n\n".join(to_text(p) for p in paragraphs)
        # else:
        #     context = ""

        rec = RouterExample(
            id=make_id("MuSiQue", ex.get("id", random.randint(0, 999999))),
            source_dataset="MuSiQue",
            question=to_text(ex.get("question", "")),
            context=str(context),
            gold_answer=to_text(ex.get("answer", "")),
            gold_label="",
            unit_tests="",
            skill_tags=["retrieval", "multi-hop"],
            hallucination_subset=False,
            split="",
            original_id=to_text(ex.get("id", "")),
            metadata= {"answerable" : ex.get("answerable", False)}
        )
        records.append(rec.to_dict())

    return records
