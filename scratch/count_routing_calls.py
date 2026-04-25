"""Count P4 component invocations per source in a results jsonl.

Reports three model-call counts per source:

  decomposer  — one call per row, except rows where the benchmark client
                set `force_single_subtask` (FEVER + the two force_role
                sources). Those rows skip decomposition entirely.
  router      — one call per subtask routed by the learned router. Rows
                with `force_role` bypass the router; their routing is
                synthetic and does not count.
  synthesizer — one call per row whose response was assembled from more
                than one specialist run. Single-run rows short-circuit
                and never touch the synthesizer.

Usage:
    uv run python scratch/count_routing_calls.py --results <file>.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


# Sources where the benchmark client bypassed the learned router via
# force_role metadata. Routing decisions for these are synthetic, not
# router invocations.
FORCE_ROLE_SOURCES = {"HARDMATH", "HumanEvalPlus"}

# Sources where the client set force_single_subtask, skipping the
# decomposer entirely (the prompt is its own single subtask).
FORCE_SINGLE_SUBTASK_SOURCES = {"FEVER", "HARDMATH", "HumanEvalPlus"}


def _routes(row: dict) -> list:
    r = row.get("predicted_route")
    if r is None:
        r = (row.get("metadata") or {}).get("predicted_route")
    if isinstance(r, list):
        return r
    return [r] if r else []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()

    per_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "rows": 0,
            "decomposer": 0,
            "router": 0,
            "synthesizer": 0,
        }
    )
    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = row.get("source_dataset") or "UNKNOWN"
            s = per_source[src]
            s["rows"] += 1

            if src not in FORCE_SINGLE_SUBTASK_SOURCES:
                s["decomposer"] += 1

            routes = _routes(row)
            if src not in FORCE_ROLE_SOURCES:
                s["router"] += len(routes)

            if len(routes) > 1:
                s["synthesizer"] += 1

    cols = ("rows", "decomposer", "router", "synthesizer")
    header = f"{'source':<16}" + "".join(f"{c:>13}" for c in cols)
    print(header)
    print("-" * len(header))

    totals = {c: 0 for c in cols}
    for src, s in sorted(per_source.items()):
        row_str = f"{src:<16}" + "".join(f"{s[c]:>13}" for c in cols)
        print(row_str)
        for c in cols:
            totals[c] += s[c]
    print("-" * len(header))
    print(f"{'ALL':<16}" + "".join(f"{totals[c]:>13}" for c in cols))


if __name__ == "__main__":
    main()
