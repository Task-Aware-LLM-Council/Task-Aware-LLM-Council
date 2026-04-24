"""Print the raw `gold_answer` shape for one row of each source_dataset.

Use when the scorer reports wildly wrong numbers (like EM=0.000 on
predictions that visibly match gold) to confirm the gold field hasn't
drifted to a new schema. Different datasets can store gold as a plain
string, a list of strings, or a list of dicts like
{"text": "...", "tokens": [...]} — `_gold_as_list` in
scratch/score_p4_results.py needs to cover all three.

Usage:
    uv run python scratch/inspect_gold_format.py \\
        --results p4_gemma_lora_v2_rd2_val.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--samples-per-source", type=int, default=2)
    args = p.parse_args()

    seen: dict[str, int] = {}
    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = row.get("source_dataset") or "UNKNOWN"
            seen[src] = seen.get(src, 0) + 1
            if seen[src] > args.samples_per_source:
                continue

            gold = row.get("gold_answer")
            label = row.get("gold_label")
            print(f"--- source={src}  idx={row.get('index')} ---")
            print(f"  pred: {(row.get('p4_answer') or '')[:120].replace(chr(10), ' / ')}")
            print(f"  gold_answer type: {type(gold).__name__}")
            if isinstance(gold, list) and gold:
                print(f"  gold_answer[0] type: {type(gold[0]).__name__}")
                print(f"  gold_answer[0] value: {gold[0]!r}")
            else:
                print(f"  gold_answer value: {gold!r}")
            if label is not None:
                print(f"  gold_label: {label!r}")
            print()


if __name__ == "__main__":
    main()
