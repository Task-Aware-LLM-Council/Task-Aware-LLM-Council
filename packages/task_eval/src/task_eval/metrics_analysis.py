import argparse
import json
from collections import defaultdict
from pathlib import Path


def find_prediction_files(suite_dir: Path) -> list[Path]:
    """
    Find all prediction jsonl files under a benchmark suite directory.

    Layout (from benchmark_runner/benchmarking_pipeline):
        suite_dir/
          predictions/
            <run_id>/
              predictions/
                <dataset>__<model>.jsonl
    """
    # Matches: suite_dir/predictions/*/predictions/*.jsonl
    return list(suite_dir.glob("predictions/*/predictions/*.jsonl"))


def load_predictions(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def group_by_example(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        ex_id = rec.get("example_id")
        if isinstance(ex_id, str) and rec.get("status") == "success":
            grouped[ex_id].append(rec)
    return grouped


def avg_tokens_per_question(grouped: dict[str, list[dict]]) -> float:
    per_example: list[float] = []
    for ex_id, calls in grouped.items():
        total_tokens = 0
        for call in calls:
            usage = call.get("usage") or {}
            # Prefer total_tokens if present, otherwise sum input+output
            tt = usage.get("total_tokens")
            if tt is None:
                tt = (usage.get("input_tokens", 0) or 0) + (
                    usage.get("output_tokens", 0) or 0
                )
            total_tokens += tt
        per_example.append(total_tokens)
    return sum(per_example) / len(per_example) if per_example else 0.0


def avg_calls_per_question(grouped: dict[str, list[dict]]) -> float:
    counts = [len(calls) for calls in grouped.values()]
    return sum(counts) / len(counts) if counts else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite-dir",
        type=str,
        required=True,
        help="Path to benchmark suite output directory (e.g., results/benchmark_suite/DeepSeek-Coder-V2-Lite-Instruct)",
    )
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    if not suite_dir.exists():
        raise SystemExit(f"suite_dir does not exist: {suite_dir}")

    pred_files = find_prediction_files(suite_dir)
    if not pred_files:
        raise SystemExit(f"No prediction files found under {suite_dir}")

    print(f"Found {len(pred_files)} prediction files under {suite_dir}\n")

    # Per-file and overall aggregates
    overall_tokens: list[float] = []
    overall_calls: list[float] = []

    for path in sorted(pred_files):
        records = load_predictions(path)
        grouped = group_by_example(records)
        avg_tok = avg_tokens_per_question(grouped)
        avg_calls = avg_calls_per_question(grouped)

        overall_tokens.append(avg_tok)
        overall_calls.append(avg_calls)

        print(f"{path}:")
        print(f"  examples: {len(grouped)}")
        print(f"  avg_tokens_per_question: {avg_tok:.2f}")
        print(f"  avg_calls_per_question: {avg_calls:.2f}\n")

    # Simple overall mean across files (you can make this more precise later)
    if overall_tokens:
        print("=== Overall (simple mean across files) ===")
        print(f"avg_tokens_per_question: {sum(overall_tokens) / len(overall_tokens):.2f}")
        print(f"avg_calls_per_question: {sum(overall_calls) / len(overall_calls):.2f}")


if __name__ == "__main__":
    main()