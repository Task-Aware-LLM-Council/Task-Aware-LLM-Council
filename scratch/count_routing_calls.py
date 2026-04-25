"""Count learned-router invocations per source in a P4 results jsonl.

Counts routing calls only — force-routed rows (force_role metadata) are
excluded from the call count because the router was bypassed.

Usage:
    uv run python scratch/count_routing_calls.py --results <file>.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


# Sources where the benchmark client bypasses the learned router via
# force_role metadata. Routing decisions for these are synthetic, not
# router invocations.
FORCE_ROLE_SOURCES = {"HARDMATH", "HumanEvalPlus"}


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
        lambda: {"rows": 0, "calls": 0, "forced": 0}
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
            if src in FORCE_ROLE_SOURCES:
                s["forced"] += 1
                continue
            s["calls"] += len(_routes(row))

    total_rows = total_calls = total_forced = 0
    print(f"{'source':<16} {'rows':>5} {'forced':>7} {'calls':>6} {'calls/row':>10}")
    print("-" * 50)
    for src, s in sorted(per_source.items()):
        routed = s["rows"] - s["forced"]
        cpr = (s["calls"] / routed) if routed else 0.0
        print(
            f"{src:<16} {s['rows']:>5} {s['forced']:>7} "
            f"{s['calls']:>6} {cpr:>10.2f}"
        )
        total_rows += s["rows"]
        total_calls += s["calls"]
        total_forced += s["forced"]
    print("-" * 50)
    routed = total_rows - total_forced
    cpr = (total_calls / routed) if routed else 0.0
    print(
        f"{'ALL':<16} {total_rows:>5} {total_forced:>7} "
        f"{total_calls:>6} {cpr:>10.2f}"
    )


if __name__ == "__main__":
    main()
