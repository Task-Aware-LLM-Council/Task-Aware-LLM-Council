"""Dump the first N HARDMATH rows where the model didn't produce a
\\boxed{} answer. Used to diagnose whether the regression is truncation
(prediction ends mid-reasoning), prompt drift (model uses different
final-answer formatting), or output-style change.
"""
import json
import re
import sys


_BOXED_RE = re.compile(r"\\boxed\s*\{")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: dump_no_pred_box.py <jsonl> [n=2]")
        sys.exit(1)

    out_path = sys.argv[1]
    n_target = int(sys.argv[2]) if len(sys.argv) >= 3 else 2

    n = 0
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("source_dataset") != "HARDMATH":
                continue
            if r.get("error"):
                continue
            pred = r.get("p4_answer") or ""
            if _BOXED_RE.search(pred):
                continue

            print(f"=== INDEX {r.get('index')}  ({len(pred)} chars) ===")
            print(f"GOLD : {str(r.get('gold_answer') or '')[:160]}")
            print("---- last 1000 chars of p4_answer ----")
            print(pred[-1000:])
            print()

            n += 1
            if n >= n_target:
                break

    if n == 0:
        print("No no_pred_box HARDMATH rows found.")


if __name__ == "__main__":
    main()
