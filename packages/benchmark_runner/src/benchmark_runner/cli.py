from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from benchmark_runner.config import get_dataset_configs, get_preset_spec
from benchmark_runner.suite import run_registered_benchmark_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark model-dataset pairs.")
    parser.add_argument("--preset", default="pilot", choices=("pilot", "full"))
    parser.add_argument(
        "--output-root",
        default="results/benchmark_suite",
        help="Directory that will contain the suite output directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset subset. Defaults to all configured datasets.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional model subset. Defaults to the configured model pool.",
    )
    parser.add_argument(
        "--sample-cap",
        type=int,
        help="Optional override for max examples per dataset.",
    )
    parser.add_argument(
        "--split",
        help="Optional override applied to all selected datasets.",
    )
    return parser


async def run_cli_async(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    spec = get_preset_spec(args.preset, output_root=output_root)

    if args.models:
        spec = replace(spec, models=tuple(args.models))
    if args.sample_cap is not None:
        spec = replace(spec, max_examples_per_dataset=args.sample_cap)

    dataset_configs = get_dataset_configs(tuple(args.datasets) if args.datasets else None)
    if args.split:
        dataset_configs = tuple(
            type(config)(name=config.name, split=args.split, profile_kwargs=dict(config.profile_kwargs))
            for config in dataset_configs
        )

    result = await run_registered_benchmark_suite(dataset_configs, spec)
    payload = {
        "suite_id": result.suite_id,
        "output_dir": str(result.output_dir),
        "aggregate_summary_path": str(result.aggregate_summary_path),
        "total_pairs": result.total_pairs,
        "total_examples": result.total_examples,
        "scored_examples": result.scored_examples,
        "failed_examples": result.failed_examples,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run_cli_async(args))
