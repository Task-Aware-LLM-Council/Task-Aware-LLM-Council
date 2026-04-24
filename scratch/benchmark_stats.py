"""Per-source aggregate stats for a benchmark run JSONL.

Reports for each source_dataset plus an overall row:
  - n, error count
  - avg latency (from `latency_ms`)
  - avg output length (chars and approximate tokens; tokens use
    `len(text)//4` as a cheap estimate — replace with a real tokenizer
    if you need sub-5% accuracy)
  - routing accuracy: fraction of rows whose `predicted_route` contains
    a role in that source's expected-role set. `force_role` rows are
    trivially 100% by design; rows dispatched through the learned
    router are what this column actually measures.

The expected-role mapping is a defensible default for this benchmark's
role vocabulary (qa_reasoning / math_code / fact_general). Override with
--expected-role if you want to score strict matches against a single
target role per source.

Usage:
    uv run python scratch/benchmark_stats.py \\
        --results p4_gemma_lora_v2_rd2_val_oracle.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


# Accept any of these roles as "correctly routed" per source. The learned
# router's output vocabulary in this repo is {qa_reasoning, math_code,
# fact_general}; a canonical expected role is listed first, with looser
# aliases as fallbacks for runs that used a different role-naming scheme.
DEFAULT_EXPECTED_ROLES: dict[str, set[str]] = {
    "FEVER":         {"fact_general"},
    "HARDMATH":      {"math_code"},
    "HumanEvalPlus": {"math_code"},
    "MuSiQue":       {"qa_reasoning"},
    "QuALITY":       {"qa_reasoning"},
}

# Sources where the benchmark client applied --force-role-sources bypass.
# Rows from these sources always show predicted_route == the forced role,
# which would inflate routing accuracy since the learned router was
# bypassed. Exclude from the route_acc column by default.
DEFAULT_FORCE_ROLE_SOURCES: frozenset[str] = frozenset({
    "HARDMATH", "HumanEvalPlus", "FEVER",
})


def _route_list(row: dict) -> list[str]:
    route = row.get("predicted_route")
    if route is None:
        route = (row.get("metadata") or {}).get("predicted_route")
    if isinstance(route, list):
        return [str(r) for r in route]
    if route:
        return [str(route)]
    return []


def _approx_tokens(text: str) -> int:
    # Cheap token estimate: ~4 chars/token for English. If you need real
    # numbers, swap in tiktoken or the model's tokenizer.
    return max(0, len(text or "")) // 4


def _mean(values):
    return sum(values) / len(values) if values else None


def _fmt(v, fmt):
    return fmt.format(v) if v is not None else "-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument(
        "--expected-roles", default=None,
        help="Optional comma-separated SOURCE:ROLE list to override the "
             "canonical mapping (e.g. FEVER:fact_general,HARDMATH:math_code). "
             "Any role in predicted_route matching ROLE counts as correct.",
    )
    p.add_argument(
        "--count-force-routed", action="store_true",
        help="Include force-routed sources (HARDMATH, HumanEvalPlus) in the "
             "route_acc column. Off by default because the learned router "
             "was bypassed for those rows — counting them inflates accuracy.",
    )
    args = p.parse_args()

    expected: dict[str, set[str]] = {
        k: set(v) for k, v in DEFAULT_EXPECTED_ROLES.items()
    }
    if args.expected_roles:
        for entry in args.expected_roles.split(","):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            src, role = entry.split(":", 1)
            expected[src.strip()] = {role.strip()}

    stats: dict[str, dict] = defaultdict(
        lambda: {
            "n": 0, "errors": 0,
            "latency_ms": [], "pred_chars": [],
            "pred_tokens": [], "correct_routes": 0, "routed_rows": 0,
        }
    )

    with args.results.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = row.get("source_dataset") or "UNKNOWN"
            s = stats[src]
            s["n"] += 1
            if row.get("error"):
                s["errors"] += 1
                continue

            lat = row.get("latency_ms")
            if isinstance(lat, (int, float)):
                s["latency_ms"].append(float(lat))

            pred = row.get("p4_answer") or ""
            s["pred_chars"].append(len(pred))
            s["pred_tokens"].append(_approx_tokens(pred))

            if args.count_force_routed or src not in DEFAULT_FORCE_ROLE_SOURCES:
                route = _route_list(row)
                if route:
                    s["routed_rows"] += 1
                    expected_set = expected.get(src, set())
                    if expected_set and any(r in expected_set for r in route):
                        s["correct_routes"] += 1

    print(f"{'source':<16} {'n':>4} {'err':>4} "
          f"{'lat_ms':>9} {'out_chars':>10} {'out_tok~':>9} "
          f"{'route_acc':>10}")
    print("-" * 70)

    totals = {
        "n": 0, "errors": 0,
        "latency_ms": [], "pred_chars": [],
        "pred_tokens": [], "correct_routes": 0, "routed_rows": 0,
    }

    for src, s in sorted(stats.items()):
        lat = _mean(s["latency_ms"])
        chars = _mean(s["pred_chars"])
        toks = _mean(s["pred_tokens"])
        acc = s["correct_routes"] / s["routed_rows"] if s["routed_rows"] else None
        print(f"{src:<16} {s['n']:>4} {s['errors']:>4} "
              f"{_fmt(lat, '{:>9.0f}')} {_fmt(chars, '{:>10.0f}')} "
              f"{_fmt(toks, '{:>9.0f}')} {_fmt(acc, '{:>10.3f}')}")

        totals["n"] += s["n"]
        totals["errors"] += s["errors"]
        totals["latency_ms"].extend(s["latency_ms"])
        totals["pred_chars"].extend(s["pred_chars"])
        totals["pred_tokens"].extend(s["pred_tokens"])
        totals["correct_routes"] += s["correct_routes"]
        totals["routed_rows"] += s["routed_rows"]

    t_lat = _mean(totals["latency_ms"])
    t_chars = _mean(totals["pred_chars"])
    t_toks = _mean(totals["pred_tokens"])
    t_acc = (
        totals["correct_routes"] / totals["routed_rows"]
        if totals["routed_rows"] else None
    )
    print("-" * 70)
    print(f"{'ALL':<16} {totals['n']:>4} {totals['errors']:>4} "
          f"{_fmt(t_lat, '{:>9.0f}')} {_fmt(t_chars, '{:>10.0f}')} "
          f"{_fmt(t_toks, '{:>9.0f}')} {_fmt(t_acc, '{:>10.3f}')}")


if __name__ == "__main__":
    main()
