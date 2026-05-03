"""Audit how many MuSiQue rows in the local router dataset have a
recoverable `is_supporting`-flagged oracle upstream — vs. how many fell
through to the silent `paragraphs[:4]` fallback in the cache builder.

The cache builder at p4_benchmark_client._build_musique_oracle_map only
distinguishes "found at least one is_supporting=True" from "fall back".
This script reports both populations so you can tell whether your eval
slice is mostly real oracle (and the regression is elsewhere) or mostly
fallback (and the cache is the regression).

Usage:
    uv run --package council_policies python scratch/oracle_quality_audit.py \\
        --router-dataset task-aware-llm-council/router_dataset-2 \\
        --router-split validation \\
        --musique-source bdsaglam/musique \\
        --musique-split validation
"""
import argparse
from collections import Counter

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--router-dataset", default="task-aware-llm-council/router_dataset-2")
    p.add_argument("--router-split", default="validation")
    p.add_argument("--musique-source", default="bdsaglam/musique")
    p.add_argument("--musique-split", default="validation")
    p.add_argument(
        "--show-fallback", type=int, default=10,
        help="print the first N (router_index, original_id) pairs that "
             "would hit the silent fallback (default 10).",
    )
    args = p.parse_args()

    print(f"Building {args.musique_source}:{args.musique_split} support index ...")
    musique = load_dataset(args.musique_source, split=args.musique_split)
    support_counts: dict[str, int] = {}
    for row in musique:
        rid = str(row.get("id", ""))
        if not rid:
            continue
        paragraphs = row.get("paragraphs") or []
        n_supp = sum(
            1 for p in paragraphs
            if isinstance(p, dict) and p.get("is_supporting") is True
        )
        support_counts[rid] = n_supp
    print(f"  indexed {len(support_counts)} musique rows")

    print(f"\nScanning {args.router_dataset}:{args.router_split} ...")
    router = load_dataset(args.router_dataset, split=args.router_split)

    buckets: Counter[str] = Counter()
    fallback_examples: list[tuple[int, str]] = []

    for i, r in enumerate(router):
        if r.get("source_dataset") != "MuSiQue":
            continue
        oid = r.get("original_id")
        if oid is None:
            buckets["no_original_id"] += 1
            continue
        rid = str(oid)
        if rid not in support_counts:
            buckets["unjoinable"] += 1
            continue
        n_supp = support_counts[rid]
        if n_supp == 0:
            buckets["fallback (paragraphs[:4])"] += 1
            if len(fallback_examples) < args.show_fallback:
                fallback_examples.append((i, rid))
        elif n_supp == 1:
            buckets["1 supporting"] += 1
        else:
            buckets[f"{n_supp} supporting"] += 1

    total = sum(buckets.values())
    print(f"\n=== oracle quality across MuSiQue subset (n={total}) ===")
    for bucket, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        share = (n / total) if total else 0.0
        print(f"  {n:4}  ({share:.3f})  {bucket}")

    real_oracle = total - buckets["fallback (paragraphs[:4])"] - buckets["no_original_id"] - buckets["unjoinable"]
    if total:
        print(
            f"\n  → {real_oracle}/{total} = {real_oracle/total:.3f} rows "
            f"got REAL oracle context"
        )
        print(
            f"  → {buckets['fallback (paragraphs[:4])']}/{total} = "
            f"{buckets['fallback (paragraphs[:4])']/total:.3f} rows "
            f"got the silent paragraphs[:4] fallback (broken oracle)"
        )

    if fallback_examples:
        print(f"\nFirst {len(fallback_examples)} fallback rows in router_dataset:")
        for i, rid in fallback_examples:
            print(f"  router_index={i}  original_id={rid}")


if __name__ == "__main__":
    main()
