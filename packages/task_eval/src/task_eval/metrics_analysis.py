"""
metrics_analysis.py

Analyzes prediction JSONL outputs from any policy (P1 suite-dir format or
P2/P3/P4 flat eval format) and produces:
  - Avg Tokens Per Question
  - Avg Calls Per Question
  - Avg Latency Per Question
  - Per-task accuracy breakdown table
  - Per-task latency breakdown table
  - Routing accuracy (P3 vs P4)
  - Accuracy-vs-cost Pareto data

Usage (single file):
  python -m task_eval.metrics_analysis --input p3 ~/Downloads/results_p3_eval_local_correct.jsonl

Usage (multi-policy comparison):
  python -m task_eval.metrics_analysis 
    --input p3 ~/Downloads/results_p3.jsonl 
    --input p4 ~/Downloads/results_p4.jsonl 
    --input baseline ~/Downloads/results_baseline.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from collections import defaultdict
from pathlib import Path


# ─── Normalization ─────────────────────────────────────────────────────────────

def _pick_primary_metric(metrics: dict, dataset: str) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    if dataset == "musique":
        return float(metrics.get("token_f1") or metrics.get("exact_match") or 0.0)
    if dataset == "quality":
        return float(metrics.get("token_f1") or metrics.get("exact_match") or 0.0)
    if dataset == "fever":
        return float(metrics.get("label_accuracy") or metrics.get("accuracy") or metrics.get("exact_match") or 0.0)
    if dataset == "hardmath":
        return float(metrics.get("math_exact_match") or metrics.get("accuracy") or metrics.get("mathexactmatch") or 0.0)
    if dataset == "humaneval":
        return float(metrics.get("pass_at_1") or metrics.get("pass@1") or 0.0)
    # Fallback
    for k in ["token_f1", "label_accuracy", "accuracy", "math_exact_match", "pass_at_1", "exact_match"]:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return 0.0

def normalize_record(rec: dict, policy: str = "UNKNOWN") -> dict:
    """
    Map P3/P4 eval JSONL fields into a unified schema.
    P1 suite-dir records are passed through with minimal changes.
    """
    # Already normalized (P1 pipeline format has 'status' and 'input_tokens')
    if "status" in rec and "input_tokens" in (rec.get("usage") or {}):
        if "policy" not in rec:
            rec["policy"] = policy
        # Latency: P1 stores latency_ms at top level
        if "wall_latency_ms" not in rec:
            rec["wall_latency_ms"] = float(rec.get("latency_ms", 0.0))
        return rec

    usage_raw = rec.get("usage") or {}

    prompt_tokens = (
        usage_raw.get("prompt_tokens")
        or usage_raw.get("prompttokens")
        or usage_raw.get("input_tokens")
        or 0
    )
    completion_tokens = (
        usage_raw.get("completion_tokens")
        or usage_raw.get("completiontokens")
        or usage_raw.get("output_tokens")
        or 0
    )
    total_tokens = (
        usage_raw.get("total_tokens")
        or usage_raw.get("totaltokens")
        or (prompt_tokens + completion_tokens)
    )

    # Latency: P3/P4 stores as latency_ms at top level
    wall_latency_ms = float(
        rec.get("wall_latency_ms")
        or rec.get("latency_ms")
        or rec.get("latencyms")
        or 0.0
    )

    failed = rec.get("failed", False)
    status = "error" if failed else "success"

    dataset = rec.get("dataset") or rec.get("dataset_name") or rec.get("task_tag") or ""

    metrics_raw = rec.get("metrics") or {}
    primary_metric = (
        rec.get("primary_metric")
        if rec.get("primary_metric") is not None
        else _pick_primary_metric(metrics_raw, dataset)
    )

    task_tag = (
        rec.get("task_tag")
        or rec.get("tasktype")
        or rec.get("task_type")
    )
    predicted_tag = (
        rec.get("predicted_tag")
        or rec.get("routedrole")
        or rec.get("routed_role")
    )

    routed_correctly = rec.get("routed_correctly")
    if routed_correctly is None and task_tag is not None and predicted_tag is not None:
        routed_correctly = (str(task_tag) == str(predicted_tag))

    reference = rec.get("reference") or {}
    gold_answers = (
        rec.get("gold_answers")
        or reference.get("answers")
        or ([reference["answer"]] if "answer" in reference else [])
        or []
    )
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    return {
        **rec,
        "policy":          policy,
        "query_id":        str(rec.get("query_id") or rec.get("example_id") or ""),
        "example_id":      str(rec.get("example_id") or rec.get("query_id") or ""),
        "dataset":         dataset,
        "status":          status,
        "primary_metric":  float(primary_metric),
        "wall_latency_ms": wall_latency_ms,
        "task_tag":        task_tag,
        "predicted_tag":   predicted_tag,
        "routed_correctly": routed_correctly,
        "gold_answers":    gold_answers,
        "final_answer":    rec.get("final_answer") or rec.get("response_text") or rec.get("responsetext") or "",
        "usage": {
            "input_tokens":  int(prompt_tokens),
            "output_tokens": int(completion_tokens),
            "total_tokens":  int(total_tokens),
        },
    }


# ─── File discovery ─────────────────────────────────────────────────────────────

def find_prediction_files(suite_dir: Path) -> list[Path]:
    """
    Find all prediction jsonl files under a P1 benchmark suite directory.
    Layout: suite_dir/predictions/<run_id>/predictions/<dataset>__<model>.jsonl
    """
    return list(suite_dir.glob("predictions/*/predictions/*.jsonl"))


# ─── Loading ────────────────────────────────────────────────────────────────────

def load_predictions(path: Path, policy: str = "UNKNOWN") -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(normalize_record(json.loads(line), policy=policy))
    return records


# ─── Grouping (for P1 suite-dir multi-call-per-example format) ─────────────────

def group_by_example(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        ex_id = rec.get("example_id")
        if isinstance(ex_id, str) and ex_id and rec.get("status") == "success":
            grouped[ex_id].append(rec)
    return grouped


# ─── Per-file efficiency metrics ────────────────────────────────────────────────

def avg_tokens_per_question(grouped: dict[str, list[dict]]) -> float:
    per_example: list[float] = []
    for calls in grouped.values():
        total = 0
        for call in calls:
            usage = call.get("usage") or {}
            tt = usage.get("total_tokens")
            if tt is None:
                tt = (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
            total += tt
        per_example.append(total)
    return sum(per_example) / len(per_example) if per_example else 0.0


def avg_calls_per_question(grouped: dict[str, list[dict]]) -> float:
    counts = [len(calls) for calls in grouped.values()]
    return sum(counts) / len(counts) if counts else 0.0


def avg_latency_per_question(records: list[dict]) -> float:
    """
    Latency is tracked at the top-level record as wall_latency_ms.
    For P3/P4 this is per-question end-to-end latency.
    For P1 suite-dir format it is per-call latency; we average across all records.
    """
    latencies = [
        r["wall_latency_ms"]
        for r in records
        if r.get("status") == "success" and r.get("wall_latency_ms", 0) > 0
    ]
    return sum(latencies) / len(latencies) if latencies else 0.0


# ─── Multi-policy aggregation tables ────────────────────────────────────────────

def accuracy_table(predictions_by_policy: dict[str, list[dict]]) -> dict:
    """
    Build a policies × datasets accuracy table.
    Returns: table[policy][dataset] = mean_primary_metric
    """
    table: dict[str, dict[str, float]] = {}
    for policy, records in predictions_by_policy.items():
        table[policy] = {}
        by_ds: dict[str, list[float]] = collections.defaultdict(list)
        for r in records:
            ds = r.get("dataset")
            if ds and r.get("status") == "success":
                by_ds[ds].append(r.get("primary_metric", 0.0))
        for ds, scores in by_ds.items():
            table[policy][ds] = round(statistics.mean(scores), 4) if scores else 0.0
    return table


def latency_table(predictions_by_policy: dict[str, list[dict]]) -> dict:
    """
    Build a policies × datasets latency table (mean wall_latency_ms per question).

    Where latency comes from:
      - P3/P4 flat JSONL: top-level 'latency_ms' field, normalized to 'wall_latency_ms'
        in normalize_record(). This is the end-to-end time for one question including
        model inference + any routing overhead.
      - P1 suite-dir: top-level 'latency_ms' per prediction record, also normalized
        to 'wall_latency_ms'.
    """
    table: dict[str, dict[str, float]] = {}
    for policy, records in predictions_by_policy.items():
        table[policy] = {}
        by_ds: dict[str, list[float]] = collections.defaultdict(list)
        for r in records:
            ds = r.get("dataset")
            lat = r.get("wall_latency_ms", 0.0)
            if ds and r.get("status") == "success" and lat > 0:
                by_ds[ds].append(lat)
        for ds, lats in by_ds.items():
            table[policy][ds] = round(statistics.mean(lats), 1) if lats else 0.0
    return table


def routing_accuracy_comparison(
    p3_records: list[dict],
    p4_records: list[dict],
) -> dict:
    """
    Compare P3 (oracle routing) vs P4 (learned routing).
    Uses task_tag (oracle) and predicted_tag (routed_role) per record.
    """
    def _ra(records: list[dict]) -> dict[str, float]:
        by_ds: dict[str, list[bool]] = collections.defaultdict(list)
        for r in records:
            if r.get("routed_correctly") is not None:
                by_ds[r["dataset"]].append(bool(r["routed_correctly"]))
        return {
            ds: round(sum(v) / len(v), 4) if v else None
            for ds, v in by_ds.items()
        }

    p3_ra = _ra(p3_records)
    p4_ra = _ra(p4_records)

    p3_map    = {r["query_id"]: r.get("predicted_tag") for r in p3_records}
    p4_map    = {r["query_id"]: r.get("predicted_tag") for r in p4_records}
    p4_oracle = {r["query_id"]: r.get("task_tag")      for r in p4_records}

    disagreements = []
    for qid, p4_tag in p4_map.items():
        if qid in p3_map and p3_map[qid] != p4_tag:
            disagreements.append({
                "query_id":   qid,
                "oracle_tag": p4_oracle.get(qid),
                "p3_tag":     p3_map[qid],
                "p4_tag":     p4_tag,
                "p4_correct": (p4_tag == p4_oracle.get(qid)),
            })

    return {
        "p3_routing_accuracy": p3_ra,
        "p4_routing_accuracy": p4_ra,
        "n_disagreements":     len(disagreements),
        "p4_wins":  sum(1 for d in disagreements if d["p4_correct"]),
        "p4_loses": sum(1 for d in disagreements if not d["p4_correct"]),
        "disagreement_sample": disagreements[:20],
    }


def pareto_data(predictions_by_policy: dict[str, list[dict]]) -> list[dict]:
    """
    Accuracy-vs-cost points for a Pareto plot.
    Returns: [{policy, avg_accuracy, avg_total_tokens, avg_latency_ms}]
    """
    points = []
    for policy, records in predictions_by_policy.items():
        ok = [r for r in records if r.get("status") == "success"]
        if not ok:
            continue
        avg_acc = statistics.mean(r.get("primary_metric", 0.0) for r in ok)
        avg_tok = statistics.mean(
            (r.get("usage") or {}).get("total_tokens", 0) for r in ok
        )
        avg_lat = statistics.mean(
            r.get("wall_latency_ms", 0.0) for r in ok if r.get("wall_latency_ms", 0) > 0
        ) if any(r.get("wall_latency_ms", 0) > 0 for r in ok) else 0.0
        points.append({
            "policy":           policy,
            "avg_accuracy":     round(avg_acc, 4),
            "avg_total_tokens": round(avg_tok, 1),
            "avg_latency_ms":   round(avg_lat, 1),
        })
    points.sort(key=lambda x: x["avg_accuracy"])
    return points


# ─── Pretty printing ─────────────────────────────────────────────────────────────

def print_summary_table(table: dict, title: str, fmt: str = ".4f") -> None:
    """Pretty-print a policies × datasets table to stdout."""
    all_ds = sorted({ds for row in table.values() for ds in row})
    header = f"{'Policy':<18}" + "".join(f"{d:>12}" for d in all_ds) + f"{'AVG':>12}"
    sep = "─" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for policy in sorted(table):
        row_vals = [table[policy].get(ds) for ds in all_ds]
        numeric  = [v for v in row_vals if v is not None]
        avg_val  = statistics.mean(numeric) if numeric else None
        row = f"{policy:<18}"
        for v in row_vals:
            row += f"{v:{fmt}}" .rjust(12) if v is not None else f"{'—':>12}"
        row += f"{avg_val:{fmt}}".rjust(12) if avg_val is not None else f"{'—':>12}"
        print(row)
    print()


def print_routing_comparison(result: dict) -> None:
    """Print routing accuracy comparison between P3 and P4."""
    print("\n=== Routing Accuracy: P3 (oracle) vs P4 (learned) ===")
    all_ds = sorted(
        set(result["p3_routing_accuracy"].keys()) |
        set(result["p4_routing_accuracy"].keys())
    )
    header = f"{'Dataset':<16}{'P3 (oracle)':>14}{'P4 (learned)':>14}"
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for ds in all_ds:
        p3v = result["p3_routing_accuracy"].get(ds)
        p4v = result["p4_routing_accuracy"].get(ds)
        p3s = f"{p3v:.4f}" if p3v is not None else "—"
        p4s = f"{p4v:.4f}" if p4v is not None else "—"
        print(f"{ds:<16}{p3s:>14}{p4s:>14}")
    print()
    print(f"  P4 disagreements with P3 oracle: {result['n_disagreements']}")
    print(f"  P4 wins  (P4 correct, P3 wrong): {result['p4_wins']}")
    print(f"  P4 loses (P4 wrong, P3 correct): {result['p4_loses']}")
    print()


def print_pareto(points: list[dict]) -> None:
    """Print accuracy-vs-cost Pareto table."""
    print("\n=== Accuracy vs Cost (Pareto) ===")
    header = f"{'Policy':<18}{'Avg Accuracy':>14}{'Avg Tokens':>12}{'Avg Latency(ms)':>17}"
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for p in points:
        print(
            f"{p['policy']:<18}"
            f"{p['avg_accuracy']:>14.4f}"
            f"{p['avg_total_tokens']:>12.1f}"
            f"{p['avg_latency_ms']:>17.1f}"
        )
    print()


# ─── Core report helper ──────────────────────────────────────────────────────────

def _report_single(label: str, records: list[dict]) -> None:
    """Print efficiency metrics for a single prediction file."""
    grouped  = group_by_example(records)
    avg_tok  = avg_tokens_per_question(grouped)
    avg_call = avg_calls_per_question(grouped)
    avg_lat  = avg_latency_per_question(records)

    print(f"\n{'─'*60}")
    print(f"Policy / file: {label}")
    print(f"  examples (successful):       {len(grouped)}")
    print(f"  avg_tokens_per_question:     {avg_tok:.2f}")
    print(f"  avg_calls_per_question:      {avg_call:.2f}")
    print(f"  avg_latency_per_question ms: {avg_lat:.2f}")   # <── latency printed here


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze policy prediction JSONL files."
    )

    # --input can be given multiple times: --input <policy_label> <path>
    parser.add_argument(
        "--input",
        nargs=2,
        metavar=("POLICY", "PATH"),
        action="append",
        help=(
            "Policy label and path to a flat predictions JSONL. "
            "Can be repeated for multi-policy comparison. "
            "Example: --input p3 ~/Downloads/results_p3.jsonl"
        ),
    )
    parser.add_argument(
        "--suite-dir",
        type=str,
        help="Path to a P1 benchmark suite output directory.",
    )
    args = parser.parse_args()

    if not args.input and not args.suite_dir:
        parser.error("Provide at least one --input or --suite-dir.")

    predictions_by_policy: dict[str, list[dict]] = {}

    # Load flat JSONL inputs (P2/P3/P4)
    if args.input:
        for policy_label, path_str in args.input:
            path = Path(path_str).expanduser()
            if not path.exists():
                raise SystemExit(f"File not found: {path}")
            records = load_predictions(path, policy=policy_label.upper())
            predictions_by_policy[policy_label.upper()] = records
            _report_single(f"{policy_label.upper()} ({path.name})", records)

    # Load P1 suite-dir inputs
    if args.suite_dir:
        suite_dir = Path(args.suite_dir)
        if not suite_dir.exists():
            raise SystemExit(f"suite_dir does not exist: {suite_dir}")
        pred_files = find_prediction_files(suite_dir)
        if not pred_files:
            raise SystemExit(f"No prediction files found under {suite_dir}")
        p1_records = []
        for p in sorted(pred_files):
            recs = load_predictions(p, policy="P1")
            p1_records.extend(recs)
            _report_single(f"P1 ({p.name})", recs)
        predictions_by_policy["P1"] = p1_records

    # Multi-policy tables (only meaningful with 2+ policies)
    if len(predictions_by_policy) >= 1:
        acc_tbl = accuracy_table(predictions_by_policy)
        lat_tbl = latency_table(predictions_by_policy)
        pareto  = pareto_data(predictions_by_policy)

        print_summary_table(acc_tbl, "=== Accuracy by Policy × Dataset ===")
        print_summary_table(lat_tbl, "=== Avg Latency (ms) by Policy × Dataset ===", fmt=".1f")
        print_pareto(pareto)

    # Routing accuracy (only if both P3 and P4 are provided)
    p3 = predictions_by_policy.get("P3")
    p4 = predictions_by_policy.get("P4")
    if p3 and p4:
        routing = routing_accuracy_comparison(p3, p4)
        print_routing_comparison(routing)
    elif p3 or p4:
        print("\n(Routing accuracy comparison requires both --input p3 ... and --input p4 ...)")


if __name__ == "__main__":
    main()