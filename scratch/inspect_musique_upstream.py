"""Inspect a single bdsaglam/musique row to see whether `is_supporting`
is exposed and what the paragraphs look like.

Used to diagnose oracle-context regressions: if no paragraph carries
`is_supporting=True`, the runner's oracle-map builder
(p4_benchmark_client._build_musique_oracle_map) silently falls back to
the first 4 paragraphs — which are NOT the gold supporting ones. That
fallback masquerades as a successful oracle injection.

Usage:
    uv run --package council_policies python scratch/inspect_musique_upstream.py \\
        --source bdsaglam/musique --split validation \\
        --target-id 2hop__810884_22402
"""
import argparse

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="bdsaglam/musique")
    p.add_argument("--split", default="validation")
    p.add_argument(
        "--target-id", required=True,
        help="row id (the value of `id`/`original_id`) to inspect.",
    )
    p.add_argument(
        "--max-paragraphs", type=int, default=10,
        help="how many paragraphs to print (default 10).",
    )
    p.add_argument(
        "--scan-all", action="store_true",
        help="Also report aggregate stats: how many rows in the split "
             "have at least one is_supporting=True paragraph.",
    )
    args = p.parse_args()

    ds = load_dataset(args.source, split=args.split)
    print(f"Loaded {args.source}:{args.split} ({len(ds)} rows)")

    target_row: dict | None = None
    n_with_support = 0

    for row in ds:
        if args.scan_all:
            paragraphs = row.get("paragraphs") or []
            if any(
                isinstance(p, dict) and p.get("is_supporting") is True
                for p in paragraphs
            ):
                n_with_support += 1

        if str(row.get("id", "")) == args.target_id:
            target_row = dict(row)
            if not args.scan_all:
                break

    if args.scan_all:
        print(
            f"Rows with ≥1 is_supporting=True paragraph: "
            f"{n_with_support}/{len(ds)} = "
            f"{n_with_support / len(ds):.3f}"
        )

    if target_row is None:
        print(f"\ntarget_id {args.target_id!r} not found.")
        raise SystemExit(1)

    paragraphs = target_row.get("paragraphs") or []
    n_supporting = sum(
        1 for p in paragraphs
        if isinstance(p, dict) and p.get("is_supporting") is True
    )

    print(f"\n=== row {args.target_id} ===")
    print(f"  question        : {target_row.get('question')!r}")
    print(f"  answer          : {target_row.get('answer')!r}")
    print(f"  total paragraphs: {len(paragraphs)}")
    print(f"  is_supporting=T : {n_supporting}")
    print(f"  paragraph keys  : "
          f"{sorted(paragraphs[0].keys()) if paragraphs and isinstance(paragraphs[0], dict) else '?'}")

    print(f"\n  first {min(args.max_paragraphs, len(paragraphs))} paragraphs:")
    for i, p in enumerate(paragraphs[:args.max_paragraphs]):
        if not isinstance(p, dict):
            print(f"    [{i}] (non-dict: {type(p).__name__})")
            continue
        flag = p.get("is_supporting")
        text = (p.get("paragraph_text") or p.get("text") or "")[:90].replace("\n", " ")
        print(f"    [{i}] is_supporting={flag!r:>5}  {text}")


if __name__ == "__main__":
    main()
