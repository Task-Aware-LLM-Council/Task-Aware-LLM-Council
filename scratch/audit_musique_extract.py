"""For every MuSiQue row in a P4 results JSONL, run the same extraction
that score_p4_results.py uses and bucket the row by extraction quality.

Token-F1 punishes extra words: "25,000 students" scores 0.67 against
gold "25000", "approximately 25,000 students" scores 0.40, and a full
sentence wrapping the answer can fall below 0.25 even when the answer
itself is correct.

Buckets:
  match            - strict EM hits one of the gold answers
  high_f1          - F1 >= 0.7 (right answer, possibly minor wording drift)
  mid_f1           - 0.3 <= F1 < 0.7 (correct answer hidden in verbose response)
  low_f1           - 0 < F1 < 0.3 (one shared token but mostly wrong/wordy)
  zero_f1          - F1 == 0 (no token overlap)
  abstention       - prediction matches an abstention sentinel (NOT FOUND)

Usage:
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
        uv run --package task_eval python scratch/audit_musique_extract.py \\
        --results p4_smoke.jsonl --show-mid 10
"""
import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

from task_eval.scoring import exact_match_multi, token_f1_multi


_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|the answer is|answer)\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_ABSTAIN_SENTINELS = frozenset({
    "NOT_FOUND", "NOT FOUND", "NOT PRESENT IN CONTEXT", "NOT PRESENT",
    "UNANSWERABLE", "CANNOT BE ANSWERED", "CANNOT ANSWER", "NO ANSWER",
})


def _strip_markdown(text: str) -> str:
    s = (text or "").strip()
    while s and s[0] in "*_":
        s = s[1:]
    while s and s[-1] in "*_":
        s = s[:-1]
    return s.strip()


def _extract_answer(response: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""
    if response.upper().rstrip(".").strip() in _ABSTAIN_SENTINELS:
        return response
    matches = _FINAL_ANSWER_RE.findall(response)
    if matches:
        return _strip_markdown(matches[-1].strip().rstrip("."))
    for line in reversed(response.splitlines()):
        stripped = line.strip()
        if stripped:
            return _strip_markdown(stripped.rstrip("."))
    return response


def _gold_as_list(gold):
    if gold is None:
        return []
    if isinstance(gold, list):
        return [str(g.get("text", g)) if isinstance(g, dict) else str(g)
                for g in gold if g]
    if isinstance(gold, dict):
        t = gold.get("text")
        return [str(t)] if t else []
    s = str(gold).strip()
    if s.startswith("[{") or s.startswith('["'):
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            return _gold_as_list(parsed)
    return [s]


def _is_abstention(pred: str) -> bool:
    cleaned = (pred or "").strip().rstrip(".").upper()
    return cleaned in _ABSTAIN_SENTINELS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--show-mid", type=int, default=10)
    p.add_argument("--show-zero", type=int, default=5)
    p.add_argument("--show-low", type=int, default=5)
    args = p.parse_args()

    buckets: Counter[str] = Counter()
    mid_examples: list[tuple[int, str, list[str], float]] = []
    low_examples: list[tuple[int, str, list[str], float]] = []
    zero_examples: list[tuple[int, str, list[str]]] = []

    with args.results.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("source_dataset") != "MuSiQue":
                continue
            if r.get("error"):
                buckets["error"] += 1
                continue

            pred_raw = r.get("p4_answer") or ""
            pred = _extract_answer(pred_raw)
            golds = _gold_as_list(r.get("gold_answer"))
            if not golds:
                buckets["no_gold"] += 1
                continue

            if _is_abstention(pred):
                buckets["abstention"] += 1
                continue

            em = exact_match_multi(pred, golds)
            f1 = token_f1_multi(pred, golds)

            if em >= 1.0:
                buckets["match"] += 1
            elif f1 >= 0.7:
                buckets["high_f1"] += 1
            elif f1 >= 0.3:
                buckets["mid_f1"] += 1
                if len(mid_examples) < args.show_mid:
                    mid_examples.append((r["index"], pred, golds, f1))
            elif f1 > 0:
                buckets["low_f1"] += 1
                if len(low_examples) < args.show_low:
                    low_examples.append((r["index"], pred, golds, f1))
            else:
                buckets["zero_f1"] += 1
                if len(zero_examples) < args.show_zero:
                    zero_examples.append((r["index"], pred, golds))

    total = sum(buckets.values())
    print(f"=== MuSiQue extraction audit (n={total}) ===")
    for bucket, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        share = (n / total) if total else 0.0
        print(f"  {n:4}  ({share:.3f})  {bucket}")

    if mid_examples:
        print(f"\n=== mid_f1 examples (correct answer hidden in verbose extraction) ===")
        for idx, pred, golds, f1 in mid_examples:
            print(f"  index {idx}  f1={f1:.2f}")
            print(f"    pred ({len(pred.split())} tokens): {pred[:200]!r}")
            print(f"    golds: {[g[:80] for g in golds]}")

    if low_examples:
        print(f"\n=== low_f1 examples ===")
        for idx, pred, golds, f1 in low_examples:
            print(f"  index {idx}  f1={f1:.2f}")
            print(f"    pred ({len(pred.split())} tokens): {pred[:200]!r}")
            print(f"    golds: {[g[:80] for g in golds]}")

    if zero_examples:
        print(f"\n=== zero_f1 examples (no token overlap) ===")
        for idx, pred, golds in zero_examples:
            print(f"  index {idx}")
            print(f"    pred: {pred[:160]!r}")
            print(f"    golds: {[g[:80] for g in golds]}")


if __name__ == "__main__":
    main()
