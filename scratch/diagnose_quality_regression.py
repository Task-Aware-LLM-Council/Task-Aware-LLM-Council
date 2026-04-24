"""Diagnose why QuALITY EM collapsed on router_dataset-2:validation.

Two hypotheses to distinguish:

1. Synth-aggregation loss — synthesizer is invoked on QuALITY rows and
   merges specialist outputs badly. Testable via synthesis_used field.

2. Specialist context truncation — QuALITY contexts exceed the spec's
   max_model_len (8192 tokens default); vLLM silently truncates from the
   start, dropping answer-bearing text. Testable via approximate token
   count of context+question.

Usage:
    uv run python scratch/diagnose_quality_regression.py \\
        --results p4_gemma_lora_v2_rd2_val.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--source", default="QuALITY",
                   help="source_dataset to filter on (default: QuALITY)")
    p.add_argument("--sample", type=int, default=5,
                   help="how many sample rows to print pred/gold for")
    p.add_argument(
        "--max-model-len", type=int, default=8192,
        help="Specialist vLLM max_model_len — rows above this are truncated.",
    )
    args = p.parse_args()

    rows = []
    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("source_dataset") == args.source:
                rows.append(row)

    if not rows:
        print(f"No {args.source} rows found in {args.results}")
        return

    print(f"=== {args.source} diagnostic ({len(rows)} rows) ===\n")

    # --- Hypothesis 1: synth path usage ---
    synth_used = Counter(r.get("synthesis_used") for r in rows)
    fallback = Counter(
        (r.get("metadata") or {}).get("synthesis_used_fallback") for r in rows
    )
    short_circuit = Counter(
        (r.get("metadata") or {}).get("synthesis_short_circuit") for r in rows
    )
    print("[H1] synth path:")
    print(f"  synthesis_used:           {dict(synth_used)}")
    print(f"  synthesis_used_fallback:  {dict(fallback)}")
    print(f"  synthesis_short_circuit:  {dict(short_circuit)}")
    multi_run = sum(1 for r in rows if r.get("synthesis_used") is True)
    print(f"  -> {multi_run}/{len(rows)} rows actually hit the synthesizer.\n")

    # --- Hypothesis 2: context truncation ---
    # ~4 chars/token is a rough rule of thumb; fine for an order-of-magnitude signal.
    over_limit = 0
    token_buckets = Counter()
    for r in rows:
        q = r.get("question") or ""
        ctx = r.get("context") or ""
        approx_tokens = (len(q) + len(ctx)) // 4
        if approx_tokens > args.max_model_len:
            over_limit += 1
        bucket = min(approx_tokens // 2000 * 2000, 12000)
        token_buckets[bucket] += 1

    print(f"[H2] context length (approx, 4 chars/token):")
    print(f"  max_model_len assumed = {args.max_model_len}")
    print(f"  rows above limit (likely truncated): {over_limit}/{len(rows)}")
    print("  distribution (ctx+question):")
    for bucket in sorted(token_buckets):
        label = f"{bucket}+" if bucket >= 12000 else f"{bucket}-{bucket + 2000}"
        print(f"    {label:>12}  tokens: {token_buckets[bucket]}")

    # --- Sample predictions vs gold ---
    print(f"\n[sample] first {args.sample} rows — pred vs gold:")
    for r in rows[: args.sample]:
        q = r.get("question") or ""
        ctx = r.get("context") or ""
        approx = (len(q) + len(ctx)) // 4
        pred = (r.get("p4_answer") or "").replace("\n", " / ")
        gold = r.get("gold_answer")
        print(f"  idx={r.get('index')}  ~{approx} tokens  "
              f"synth_used={r.get('synthesis_used')}")
        print(f"    pred: {pred[:200]}")
        print(f"    gold: {gold}")
        print()

    # --- Verdict ---
    print("=== verdict ===")
    if multi_run >= len(rows) // 2:
        print("H1 plausible: synth fires on a majority of rows — investigate "
              "synth prompt construction (build_ordered_synthesis_prompt).")
    else:
        print(f"H1 unlikely: only {multi_run}/{len(rows)} rows hit synth. "
              f"The regression is from the specialist itself.")

    if over_limit >= len(rows) // 4:
        print(f"H2 likely: {over_limit} rows exceed max_model_len — silent "
              f"truncation plausible. Consider raising max_model_len or "
              f"switching QuALITY to a truncation-aware template.")
    else:
        print(f"H2 unlikely: only {over_limit} rows exceed max_model_len. "
              f"Regression is from something else (extraction, prompt, "
              f"dataset drift).")


if __name__ == "__main__":
    main()
