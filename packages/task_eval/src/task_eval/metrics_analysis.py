"""
metrics_analysis.py — print a benchmark suite summary in the standard table format.

Usage:
    python -m task_eval.metrics_analysis --suite-dir /scratch1/.../suite_ID --policy P2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DATASET_SHORT: dict[str, str] = {
    "musique": "musique",
    "quality": "quality",
    "fever": "fever",
    "hardmath": "hardmath",
    "humaneval_plus": "humaneval",
}
DATASET_ORDER = ["fever", "hardmath", "humaneval", "musique", "quality"]


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility (imported by __init__.py)
# ---------------------------------------------------------------------------

def find_prediction_files(suite_dir: Path) -> list[Path]:
    return list(suite_dir.glob("predictions/*/predictions/*.jsonl"))


def load_predictions(path: Path) -> list[dict]:
    return _read_jsonl(path)


def group_by_example(records: list[dict]) -> dict[str, list[dict]]:
    from collections import defaultdict as _dd
    grouped: dict[str, list[dict]] = _dd(list)
    for rec in records:
        ex_id = rec.get("example_id")
        if isinstance(ex_id, str) and rec.get("status") == "success":
            grouped[ex_id].append(rec)
    return grouped


def avg_tokens_per_question(grouped: dict[str, list[dict]]) -> float:
    per: list[float] = []
    for calls in grouped.values():
        total = 0
        for call in calls:
            usage = call.get("usage") or {}
            tt = usage.get("total_tokens") or (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
            total += tt
        per.append(total)
    return sum(per) / len(per) if per else 0.0


def avg_calls_per_question(grouped: dict[str, list[dict]]) -> float:
    counts = [len(v) for v in grouped.values()]
    return sum(counts) / len(counts) if counts else 0.0


# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Print benchmark suite metrics in standard table format.")
    parser.add_argument("--suite-dir", required=True, help="Path to benchmark suite output directory")
    parser.add_argument("--policy", default="P2", help="Policy label shown in the table (e.g. P1, P2)")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    if not suite_dir.exists():
        raise SystemExit(f"suite-dir does not exist: {suite_dir}")

    # --- load score records (have: example_id, source_dataset, metric_name, metrics) ---
    score_by_example: dict[str, dict] = {}
    for path in sorted(suite_dir.glob("scores/*.jsonl")):
        for rec in _read_jsonl(path):
            if rec.get("status") == "scored":
                score_by_example[rec["example_id"]] = rec

    # --- load prediction records (have: example_id, usage, latency_ms) ---
    pred_by_example: dict[str, dict] = {}
    for path in sorted(suite_dir.glob("predictions/*/predictions/*.jsonl")):
        for rec in _read_jsonl(path):
            if rec.get("status") == "success":
                eid = rec.get("example_id")
                if eid:
                    pred_by_example[eid] = rec

    if not score_by_example:
        raise SystemExit(f"No scored examples found under {suite_dir}/scores/")

    # --- aggregate per dataset ---
    dataset_scores:  dict[str, list[float]] = defaultdict(list)
    dataset_tokens:  dict[str, list[float]] = defaultdict(list)
    dataset_latency: dict[str, list[float]] = defaultdict(list)
    all_tokens:  list[float] = []
    all_latency: list[float] = []

    for eid, score_rec in score_by_example.items():
        source = score_rec.get("example_metadata", {}).get("source_dataset", "unknown")
        ds = DATASET_SHORT.get(source, source)

        metric_name = score_rec.get("metric_name") or "token_f1"
        score = score_rec.get("metrics", {}).get(metric_name, 0.0) or 0.0
        dataset_scores[ds].append(score)

        pred = pred_by_example.get(eid, {})
        usage = pred.get("usage") or {}
        tokens = float(
            usage.get("total_tokens")
            or (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
        )
        dataset_tokens[ds].append(tokens)
        all_tokens.append(tokens)

        lat = pred.get("latency_ms")
        if lat is not None:
            dataset_latency[ds].append(float(lat))
            all_latency.append(float(lat))

    n = len(score_by_example)
    avg_tok = sum(all_tokens) / n if n else 0.0
    avg_lat = sum(all_latency) / len(all_latency) if all_latency else 0.0
    # P2 makes 9 LLM calls internally per example (3 answer + 3 vote + 1 synthesis + retries)
    # P1 makes 1 call. Detect by checking if any pred has usage > ~1 call worth.
    avg_calls = _estimate_calls(pred_by_example)

    # --- print ---
    sep = "_" * 52
    print(f"\n{sep}\n")
    print(f"Policy / file: {args.policy} ({suite_dir.name})")
    print(f"examples (successful):        {n}")
    print(f"avg_tokens_per_question:     {avg_tok:.2f}")
    print(f"avg_calls_per_question:      {avg_calls:.2f}")
    print(f"avg_latency_per_question ms: {avg_lat:.2f}")

    policy_w = 14
    # column width = dataset name length + 3 padding (matches P3 format)
    col_widths = {d: len(d) + 3 for d in DATASET_ORDER}

    def _header() -> str:
        return f"{'Policy':<{policy_w}}" + "".join(f"{d:<{col_widths[d]}}" for d in DATASET_ORDER) + "AVG"

    def _divider() -> str:
        total = policy_w + sum(col_widths[d] for d in DATASET_ORDER) + 3
        return "-" * total

    # accuracy table
    print(f"\n=== Accuracy by Policy × Dataset ===\n")
    print(_header())
    print(_divider())
    accs: list[float] = []
    row = f"{args.policy:<{policy_w}}"
    for d in DATASET_ORDER:
        cw = col_widths[d]
        scores = dataset_scores.get(d, [])
        if scores:
            v = sum(scores) / len(scores)
            accs.append(v)
            row += f"{v:<{cw}.4f}"
        else:
            row += f"{'n/a':<{cw}}"
    avg_acc = sum(accs) / len(accs) if accs else 0.0
    print(row + f"{avg_acc:.4f}")

    # latency table
    print(f"\n=== Avg Latency (ms) by Policy × Dataset ===\n")
    print(_header())
    print(_divider())
    lats: list[float] = []
    row = f"{args.policy:<{policy_w}}"
    for d in DATASET_ORDER:
        cw = col_widths[d]
        vals = dataset_latency.get(d, [])
        if vals:
            v = sum(vals) / len(vals)
            lats.append(v)
            row += f"{v:<{cw}.1f}"
        else:
            row += f"{'n/a':<{cw}}"
    avg_lat_all = sum(lats) / len(lats) if lats else avg_lat
    print(row + f"{avg_lat_all:.1f}")

    # pareto table
    print(f"\n=== Accuracy vs Cost (Pareto) ===\n")
    print(f"{'Policy':<{policy_w}}{'Avg Accuracy':<16}{'Avg Tokens':<14}{'Avg Latency(ms)'}")
    print("-" * 57)
    print(f"{args.policy:<{policy_w}}{avg_acc:<16.4f}{avg_tok:<14.1f}{avg_lat:.1f}")
    print()


def _estimate_calls(pred_by_example: dict[str, dict]) -> float:
    """Guess avg LLM calls per example from metadata if available, else return 1."""
    calls = []
    for pred in pred_by_example.values():
        n = (pred.get("request_metadata") or {}).get("num_llm_calls")
        if n is not None:
            calls.append(float(n))
    return sum(calls) / len(calls) if calls else 1.0


if __name__ == "__main__":
    main()
