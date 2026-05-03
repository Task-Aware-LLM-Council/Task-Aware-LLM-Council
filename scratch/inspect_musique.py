"""Quick MuSiQue slice inspector for a P4 results JSONL.

Surfaces the three signals you usually want first when MuSiQue scores
look off:
  1. route distribution — how often the joint router sent MuSiQue rows
     to qa_reasoning vs other specialists, separated from `force_role`
     rows whose routing was synthetic.
  2. synthesis_used breakdown — whether multi-specialist synthesis is
     firing on rows that should have short-circuited (or vice versa).
  3. gold-vs-pred samples — eyeball the first N rows so you can tell
     extraction artifacts apart from genuine model misses.

Usage:
    uv run --package task_eval python scratch/inspect_musique.py \\
        --results p4_smoke.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument(
        "--n", type=int, default=10,
        help="sample rows to print (default 10)",
    )
    args = p.parse_args()

    routes: Counter[tuple] = Counter()
    synth: Counter[bool] = Counter()
    samples: list[dict] = []
    n_total = 0
    n_errors = 0

    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("source_dataset") != "MuSiQue":
                continue
            n_total += 1
            if r.get("error"):
                n_errors += 1
                continue

            route = tuple(r.get("predicted_route") or [])
            forced = r.get("force_role") or "free"
            routes[(route, forced)] += 1
            synth[bool(r.get("synthesis_used"))] += 1

            if len(samples) < args.n:
                samples.append(r)

    print(f"MuSiQue rows: {n_total}  (errors: {n_errors})")

    print("\n=== route distribution ===")
    for (route, forced), n in routes.most_common():
        print(f"  {n:4}  route={route}  force={forced}")

    print("\n=== synthesis_used ===")
    for k, n in synth.items():
        print(f"  {str(k):<5} {n}")

    print(f"\n=== first {len(samples)} rows (gold vs pred) ===")
    for r in samples:
        pred = (r.get("p4_answer") or "").replace("\n", " ")[:200]
        gold = str(r.get("gold_answer"))[:80]
        print(f"[{r['index']}]")
        print(f"  GOLD: {gold}")
        print(f"  PRED: {pred}")


if __name__ == "__main__":
    main()
