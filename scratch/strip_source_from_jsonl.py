"""Drop all rows of a given source_dataset from a benchmark jsonl in place.

Used to re-run a subset of rows after a code change (e.g. regenerate
HumanEvalPlus rows with the new force-role + template fixes without
re-running the full benchmark). The client's resume logic then skips
the preserved rows and only redoes the dropped ones.

The input file is overwritten — the dropped rows are lost. If you want
a backup, cp the file first.

Usage:
    uv run python scratch/strip_source_from_jsonl.py \\
        --results p4_gemma_lora_v2_rd2_val_oracle.jsonl \\
        --source HumanEvalPlus
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--source", required=True,
                   help="source_dataset value to drop (e.g. HumanEvalPlus).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report counts only, don't rewrite the file.")
    args = p.parse_args()

    rows: list[dict] = []
    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    before = Counter(r.get("source_dataset") for r in rows)
    kept = [r for r in rows if r.get("source_dataset") != args.source]
    dropped = len(rows) - len(kept)
    after = Counter(r.get("source_dataset") for r in kept)

    print(f"Before: {dict(before)}")
    print(f"After:  {dict(after)}")
    print(f"Dropped {dropped} row(s) with source_dataset={args.source!r}")

    if args.dry_run:
        print("[dry-run] not rewriting file.")
        return

    if dropped == 0:
        print("Nothing to do.")
        return

    with args.results.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(kept)} rows back to {args.results}")


if __name__ == "__main__":
    main()
