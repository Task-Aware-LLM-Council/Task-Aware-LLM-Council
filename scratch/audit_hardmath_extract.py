"""For every HARDMATH row in a P4 results JSONL, run task_eval's
extract_math_answer + math_exact_match and bucket the row into one of:

  match            — pred and gold extract to the same string and match
  format_mismatch  — both extracted, model is correct under loose match,
                     scorer says wrong (whitespace, bare-vs-prefixed, etc.)
  no_pred_box      — pred has no \\boxed{} (model didn't follow the prompt)
  no_gold_box      — gold has no \\boxed{} (dataset row is malformed)
  genuine_miss     — both extracted, fundamentally different answers

Usage:
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        uv run --package task_eval python scratch/audit_hardmath_extract.py \\
        --results p4_smoke.jsonl --show-format-mismatch 10
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from task_eval.extraction import extract_math_answer
from task_eval.scoring import math_exact_match


_BOXED_RE = re.compile(r"\\boxed\s*\{")


def _has_boxed(text: str) -> bool:
    return bool(_BOXED_RE.search(text or ""))


def _loose_norm(s: str) -> str:
    """Whitespace + dollar sign + simple latex prefix stripping. Used only
    to label rows as 'format_mismatch' (would-have-passed under loose
    matching), not for actual scoring."""
    s = (s or "").strip()
    s = s.replace("$", "")
    s = re.sub(r"\s+", "", s)
    # strip common scoping prefixes like "x \in", "x =", etc
    s = re.sub(r"^[a-zA-Z]\\?in", "", s)
    s = re.sub(r"^[a-zA-Z]=", "", s)
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument(
        "--show-format-mismatch", type=int, default=10,
        help="Print up to N rows where strict EM=0 but loose EM=1.",
    )
    p.add_argument(
        "--show-genuine-miss", type=int, default=5,
        help="Print up to N genuine-miss rows for sanity.",
    )
    args = p.parse_args()

    buckets: Counter[str] = Counter()
    format_mismatches: list[tuple[int, str, str]] = []
    genuine_misses: list[tuple[int, str, str]] = []

    with args.results.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("source_dataset") != "HARDMATH":
                continue
            if r.get("error"):
                buckets["error"] += 1
                continue

            pred_raw = r.get("p4_answer") or ""
            gold_raw = str(r.get("gold_answer") or "")

            pred_has = _has_boxed(pred_raw)
            gold_has = _has_boxed(gold_raw)

            pred = extract_math_answer(pred_raw)
            gold = extract_math_answer(gold_raw)

            strict = math_exact_match(pred, gold)

            if not pred_has:
                buckets["no_pred_box"] += 1
                continue
            if not gold_has:
                buckets["no_gold_box"] += 1
                continue
            if strict:
                buckets["match"] += 1
                continue

            # Both boxed, strict match failed. Classify further.
            if _loose_norm(pred) == _loose_norm(gold):
                buckets["format_mismatch"] += 1
                if len(format_mismatches) < args.show_format_mismatch:
                    format_mismatches.append((r["index"], pred, gold))
            else:
                buckets["genuine_miss"] += 1
                if len(genuine_misses) < args.show_genuine_miss:
                    genuine_misses.append((r["index"], pred, gold))

    total = sum(buckets.values())
    print(f"=== HARDMATH extraction audit (n={total}) ===")
    for bucket, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        share = (n / total) if total else 0.0
        print(f"  {n:4}  ({share:.3f})  {bucket}")

    if format_mismatches:
        print(f"\n=== format_mismatch examples (would pass with loose normalization) ===")
        for idx, pred, gold in format_mismatches:
            print(f"  index {idx}")
            print(f"    pred: {pred!r}")
            print(f"    gold: {gold!r}")

    if genuine_misses:
        print(f"\n=== genuine_miss examples ===")
        for idx, pred, gold in genuine_misses:
            print(f"  index {idx}")
            print(f"    pred: {pred!r}")
            print(f"    gold: {gold!r}")


if __name__ == "__main__":
    main()
