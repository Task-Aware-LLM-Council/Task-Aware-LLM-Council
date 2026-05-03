"""Probe the MuSiQue oracle-context join used by p4_benchmark_client.py.

The runner injects oracle context only when `row["original_id"]` resolves
in the cached `musique_oracle_map_*.json`. If the join misses, the
specialist sees router_dataset's distractor-full default context instead
and MuSiQue scores silently regress.

This script answers three questions:
  1. Does the row at --row-index resolve in the map? (debugging one
     specific JSONL entry)
  2. What fraction of MuSiQue rows in the split actually join?
  3. For the rows that miss, do they have a different id-like field?
     (tells us whether the fix is a multi-key fallback in
     `build_requests()` or a router_dataset regen.)

Usage:
    uv run --package council_policies python scratch/check_musique_oracle_join.py \\
        --dataset task-aware-llm-council/router_dataset-2 \\
        --split validation \\
        --oracle-map artifacts/musique_oracle_map_bdsaglam__musique_validation.json \\
        --row-index 0
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


_ID_LIKE = ("original_id", "id", "musique_id", "example_id")


def _id_like_fields(row: dict) -> dict:
    return {k: row[k] for k in row if "id" in k.lower()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="task-aware-llm-council/router_dataset-2")
    p.add_argument("--split", default="validation")
    p.add_argument(
        "--oracle-map", type=Path,
        default=Path("artifacts/musique_oracle_map_bdsaglam__musique_validation.json"),
    )
    p.add_argument(
        "--row-index", type=int, default=0,
        help="Specific dataset row index to inspect (default 0).",
    )
    p.add_argument(
        "--show-misses", type=int, default=5,
        help="Print id-like fields for the first N missing MuSiQue rows.",
    )
    args = p.parse_args()

    if not args.oracle_map.exists():
        print(f"Oracle map not found at {args.oracle_map}")
        print("Build it by running the benchmark with --musique-oracle (default).")
        raise SystemExit(1)

    oracle = json.loads(args.oracle_map.read_text(encoding="utf-8"))
    print(f"Oracle map: {len(oracle)} entries from {args.oracle_map}")

    ds = load_dataset(args.dataset, split=args.split)
    print(f"Dataset: {args.dataset}:{args.split} ({len(ds)} rows)")

    # ---- (1) Single-row probe ------------------------------------------------
    target = args.row_index
    if 0 <= target < len(ds):
        r = ds[target]
        print(f"\n=== row {target} ===")
        print(f"  source_dataset : {r.get('source_dataset')}")
        print(f"  id-like fields : {_id_like_fields(r)}")
        if r.get("source_dataset") == "MuSiQue":
            primary = r.get("original_id")
            print(f"  original_id    : {primary!r}")
            print(f"  in oracle map  : {str(primary) in oracle}")
            # also try fallback keys to see if a different field would have matched
            fallback_hits = [
                k for k in _ID_LIKE
                if k != "original_id" and str(r.get(k)) in oracle
            ]
            if fallback_hits:
                print(f"  fallback keys that WOULD have hit: {fallback_hits}")
    else:
        print(f"\nrow {target} out of bounds for {args.dataset}:{args.split}")

    # ---- (2) Aggregate join rate ---------------------------------------------
    hits = misses = 0
    sample_misses: list[tuple[int, dict]] = []
    for i, r in enumerate(ds):
        if r.get("source_dataset") != "MuSiQue":
            continue
        oid = r.get("original_id")
        if oid is not None and str(oid) in oracle:
            hits += 1
        else:
            misses += 1
            if len(sample_misses) < args.show_misses:
                sample_misses.append((i, _id_like_fields(r)))

    total = hits + misses
    rate = (hits / total) if total else 0.0
    print(
        f"\n=== aggregate ===\n"
        f"MuSiQue oracle join rate: {hits}/{total} = {rate:.3f}  "
        f"({misses} misses)"
    )

    # ---- (3) Diagnostics for misses -----------------------------------------
    if sample_misses:
        print(f"\nFirst {len(sample_misses)} missing rows (id-like fields):")
        for i, fields in sample_misses:
            print(f"  index {i}: {fields}")

        any_fallback = False
        for i, fields in sample_misses:
            for k, v in fields.items():
                if k == "original_id":
                    continue
                if v is not None and str(v) in oracle:
                    print(
                        f"  → row {i} would join via fallback key {k!r} (value {v!r})"
                    )
                    any_fallback = True
                    break

        if any_fallback:
            print(
                "\nFix suggestion: extend the runner's oracle lookup at "
                "p4_benchmark_client.py:384 to fall through `original_id`, "
                "`id`, `musique_id` (mirror score_p4_results._ORIGINAL_ID_KEYS)."
            )
        else:
            print(
                "\nNo fallback id-like field resolves either. The router_dataset "
                "MuSiQue rows do not carry the upstream MuSiQue id at all — "
                "you'll need a router_dataset regen, or a different join key."
            )


if __name__ == "__main__":
    main()
