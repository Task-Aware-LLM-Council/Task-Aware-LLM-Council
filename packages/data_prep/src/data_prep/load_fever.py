import random
from datasets import load_dataset
from common.ids import make_id
from common.schema import RouterExample


NUM_LABELS = 3
MIN_CLAIM_LENGTH = 20


def load_fever(n: int = 40, seed: int = 42) -> list[dict]:
    print(f"Loading FEVER ({n} samples)...")
    ds = load_dataset(
        "copenlu/fever_gold_evidence",
        split="validation",
        streaming=True
    )

    ds = ds.shuffle(seed=seed, buffer_size=1000)

    # FEVER has 3 main labels. Set a cap to ensure a roughly balanced sample.
    label_cap = (n // NUM_LABELS) + 1
    label_counts = {}
    sampled = []

    for ex in ds:
        claim = ex.get("claim", "")
        if len(claim) > MIN_CLAIM_LENGTH:
            lbl = ex.get("label", "")
            count = label_counts.get(lbl, 0)

            # Only add if we haven't hit the cap for this specific label
            if count < label_cap and len(sampled) < n:
                sampled.append(ex)
                label_counts[lbl] = count + 1

        if len(sampled) >= n:
            break

    records = []
    for ex in sampled:
        evidence = ex.get("evidence", "")
        if isinstance(evidence, list):
            evidence_text = "\n".join(
                item[2] if isinstance(item, list) and len(
                    item) > 2 else str(item)
                for item in evidence
            )
        else:
            evidence_text = str(evidence)

        claim = ex.get("claim", "")
        question = (
            f"Is it true that {claim[0].lower() + claim[1:]}" if claim else ""
        )
        if question and not question.endswith("?"):
            question = question.rstrip(".") + "?"
        rec = RouterExample(
            id=make_id("FEVER", ex.get("id", ex.get(
                "original_id", random.randint(0, 999999)))),
            source_dataset="FEVER",
            question=question,
            context=evidence_text,
            gold_answer=ex.get("label", ""),
            gold_label=ex.get("label", ""),
            skill_tags=["fact-check"],
            original_id=str(ex.get("original_id", ex.get("id", ""))),
        )
        records.append(rec.to_dict())

    return records
