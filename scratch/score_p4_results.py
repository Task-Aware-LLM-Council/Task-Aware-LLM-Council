"""Score a P4 benchmark JSONL using P3's compute_metrics logic.

Mirrors the P3 `council_policies.cli:compute_metrics` scoring path as
described in the session handoff:
  - _extract_answer priority: (1) `Final Answer:` regex, (2) last non-empty
    line for musique|quality|fever, (3) full stripped response.
  - FEVER       : label_accuracy (whole-word SUPPORT/REFUTE/NOT ENOUGH INFO).
  - HARDMATH    : `\\boxed{}` extraction on both pred AND gold, then EM.
  - MuSiQue     : exact_match_multi + token_f1_multi (best-of-golds).
  - QuALITY     : exact_match_multi + token_f1_multi.
  - NOT_FOUND abstention: preserved but NOT rewarded here — router_dataset
    doesn't carry the `answerable` flag needed for P3's MuSiQue abstention
    bonus. See handoff doc.

HumanEvalPlus pass@1 is scored separately via score_humaneval_pass1.py.

Usage:
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        uv run --package task_eval python scratch/score_p4_results.py \\
        --results p4_gemma_lora_v2.jsonl
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from task_eval.extraction import extract_math_answer
from task_eval.scoring import (
    exact_match_multi,
    label_accuracy,
    math_exact_match,
    token_f1_multi,
)


_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|the answer is|answer)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_LAST_LINE_SOURCES = {"MuSiQue", "QuALITY", "FEVER"}


def _extract_answer(response: str, source: str) -> str:
    """P3-style priority extraction. Preserves NOT_FOUND sentinel."""
    response = (response or "").strip()
    if not response:
        return ""
    if response == "NOT_FOUND":
        return "NOT_FOUND"

    matches = _FINAL_ANSWER_RE.findall(response)
    if matches:
        return matches[-1].strip().rstrip(".")

    if source in _LAST_LINE_SOURCES:
        for line in reversed(response.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped.rstrip(".")

    return response


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
            if source == "HumanEvalPlus":
                # Scored separately via score_humaneval_pass1.py (execution).
                continue
            per_source[source]["n"] += 1
            if row.get("error"):
                per_source[source]["errors"] += 1
                continue

            pred_raw = row.get("p4_answer") or ""

            if source == "HARDMATH":
                pred = extract_math_answer(pred_raw)
            else:
                pred = _extract_answer(pred_raw, source)

            gold_field = "gold_label" if source == "FEVER" else "gold_answer"
            gold = _gold_as_list(row.get(gold_field))
            if not gold:
                continue

            if source == "FEVER":
                per_source[source]["acc"].append(label_accuracy(pred, gold[0]))
            elif source == "HARDMATH":
                gold_extracted = extract_math_answer(gold[0])
                per_source[source]["acc"].append(
                    math_exact_match(pred, gold_extracted)
                )
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
