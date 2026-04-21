"""Score a P4 benchmark JSONL, broken down by source_dataset.

Usage:
    uv run --package council_policies python scratch/score_p4_results.py \\
        --results p4_gemma_lora_v2.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from task_eval.extraction import (
    extract_fever_label,
    extract_math_answer,
    extract_qa_answer,
    extract_qa_answer_musique,
)
from task_eval.scoring import (
    exact_match_multi,
    label_accuracy,
    math_exact_match,
    token_f1_multi,
)


EXTRACTORS = {
    "MuSiQue": extract_qa_answer_musique,
    "HotpotQA": extract_qa_answer,
    "2WikiMultiHopQA": extract_qa_answer,
    "QuALITY": extract_qa_answer,
    "FEVER": extract_fever_label,
    "HARDMATH": extract_math_answer,
}


def _gold_as_list(gold):
    if gold is None:
        return []
    if isinstance(gold, list):
        return [str(g) for g in gold if g]
    return [str(gold)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()

    per_source = defaultdict(
        lambda: {"em": [], "f1": [], "acc": [], "n": 0, "errors": 0}
    )

    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = row.get("source_dataset") or "UNKNOWN"
            per_source[source]["n"] += 1
            if row.get("error"):
                per_source[source]["errors"] += 1
                continue

            extractor = EXTRACTORS.get(source, extract_qa_answer)
            pred_raw = row.get("p4_answer") or ""
            pred = extractor(pred_raw)

            gold_field = "gold_answer"
            if source == "FEVER":
                gold_field = "gold_label"
            gold = _gold_as_list(row.get(gold_field))

            if not gold:
                continue

            if source == "FEVER":
                per_source[source]["acc"].append(label_accuracy(pred, gold[0]))
            elif source == "HARDMATH":
                gold_extracted = extract_math_answer(gold[0])
                per_source[source]["acc"].append(math_exact_match(pred, gold_extracted))
            else:
                per_source[source]["em"].append(exact_match_multi(pred, gold))
                per_source[source]["f1"].append(token_f1_multi(pred, gold))

    print(f"{'source':<20} {'n':>4} {'err':>4} {'EM':>8} {'F1':>8} {'Acc':>8}")
    print("-" * 58)
    for source, stats in sorted(per_source.items()):
        n = stats["n"]
        em = sum(stats["em"]) / len(stats["em"]) if stats["em"] else None
        f1 = sum(stats["f1"]) / len(stats["f1"]) if stats["f1"] else None
        acc = sum(stats["acc"]) / len(stats["acc"]) if stats["acc"] else None
        em_s = f"{em:>8.3f}" if em is not None else f"{'-':>8}"
        f1_s = f"{f1:>8.3f}" if f1 is not None else f"{'-':>8}"
        acc_s = f"{acc:>8.3f}" if acc is not None else f"{'-':>8}"
        print(f"{source:<20} {n:>4} {stats['errors']:>4} {em_s} {f1_s} {acc_s}")


if __name__ == "__main__":
    main()
