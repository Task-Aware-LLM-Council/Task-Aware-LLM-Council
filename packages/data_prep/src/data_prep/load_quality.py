import random
from datasets import load_dataset

from common.ids import make_id
from common.schema import RouterExample
from common.utils import to_text


def load_quality(n: int = 30, seed: int = 42) -> list[dict]:
    print(f"Loading QuALITY / NarrativeQA ({n} samples)...")

    # Enable streaming to bypass the disk download
    ds = load_dataset(
        "narrativeqa",
        split="validation",
        streaming=True
    )

    # Maintain a rolling buffer to randomize the incoming stream
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    # Keep only items with usable context
    sampled = []
    for ex in ds:
        doc = ex.get("document", {})
        summary_text = doc.get("summary", {}).get("text", "")

        if summary_text:
            sampled.append(ex)

        if len(sampled) >= n:
            break

    records = []
    for ex in sampled:
        doc = ex.get("document", {})
        qobj = ex.get("question", {})

        if isinstance(qobj, dict):
            question = to_text(qobj.get("text", ""))
        else:
            question = to_text(qobj)

        summary = doc.get("summary", {})
        if isinstance(summary, dict):
            context = to_text(summary.get("text", ""))
        else:
            context = to_text(summary)

        answers = ex.get("answers", [])
        if answers:
            if isinstance(answers[0], dict):
                gold_answer = to_text(answers[0].get("text", ""))
            else:
                gold_answer = to_text(answers[0])
        else:
            gold_answer = ""

        rec = RouterExample(
            id=make_id("QuALITY", doc.get("id", random.randint(0, 999999))),
            source_dataset="QuALITY",
            question=question,
            context=context,
            gold_answer=gold_answer,
            gold_label="",
            unit_tests="",
            skill_tags=["long-context"],
            hallucination_subset=False,
            split="",
            original_id=to_text(doc.get("id", "")),
        )
        records.append(rec.to_dict())

    return records
