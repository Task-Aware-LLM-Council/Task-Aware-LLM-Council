"""Dump the first N successful HARDMATH rows from a P4 results JSONL.

Use to inspect what the math_code specialist (DeepSeek-R1) is actually
producing — whether reasoning ends with a parseable `\\boxed{...}`, whether
the model truncates, or whether the output style has drifted from the
math-specialist baseline.

Usage:
    python3 scratch/dump_hardmath.py p4_smoke.jsonl
    python3 scratch/dump_hardmath.py p4_smoke.jsonl 3        # 3 samples
"""
import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: dump_hardmath.py <jsonl> [n=2]")
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

            print(f"=== INDEX {r.get('index')} ===")
            print(f"ROUTE: {r.get('predicted_route')}  force: {r.get('force_role')}")
            gold = str(r.get("gold_answer") or "")[:200]
            print(f"GOLD : {gold}")
            print("---- p4_answer (full) ----")
            print(r.get("p4_answer") or "")
            print()

            n += 1
            if n >= n_target:
                break

    if n == 0:
        print("No successful HARDMATH rows found.")


if __name__ == "__main__":
    main()
