from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from common import get_current_user

from council_policies.models import P2RunConfig
from council_policies.p2.run import (
    P2_DEFAULT_DATASET,
    P2_DEFAULT_DATASET_ALIAS,
    run_p2_suite,
)

_PRESET_DEFAULTS = {
    "pilot": {"max_examples": 5, "max_concurrency": 1, "batch_size": 1},
    "full": {"max_examples": 160, "max_concurrency": 5, "batch_size": 64},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run council policies.")
    parser.add_argument("--policy", choices=("p2",), default="p2")
    parser.add_argument("--preset", default="pilot", choices=("pilot", "full"))
    parser.add_argument(
        "--output-root",
        default=f"/scratch1/{get_current_user()}/results/council_policies",
        help="Directory that will contain the P2 run directory.",
    )
    parser.add_argument(
        "--dataset",
        default=P2_DEFAULT_DATASET,
        help="Dataset name for the isolated mixed P2 dataset.",
    )
    parser.add_argument(
        "--dataset-alias",
        default=P2_DEFAULT_DATASET_ALIAS,
        help="Short dataset label stored in run artifacts.",
    )
    parser.add_argument("--sample-cap", type=int, help="Optional override for max examples.")
    parser.add_argument("--batch-size", type=int, help="Optional override for rows processed per batch.")
    parser.add_argument("--max-concurrency", type=int, help="Optional override for row concurrency.")
    parser.add_argument("--split", help="Optional dataset split override.")
    parser.add_argument(
        "--provider",
        choices=("local", "vllm", "huggingface", "openai", "openrouter", "openai-compatible"),
        help="Optional provider override. Use `vllm` for local vLLM orchestration.",
    )
    parser.add_argument("--api-base", help="Optional provider endpoint override.")
    parser.add_argument("--local-launch-port", type=int, default=8000, help="Base port for local vLLM specialist servers.")
    parser.add_argument(
        "--gpu-utilization",
        type=float,
        default=None,
        help=(
            "GPU memory utilization fraction for local vLLM servers. "
            "Defaults to 0.3."
        ),
    )
    parser.add_argument("--api-key-env", help="Optional environment variable name for provider auth.")
    parser.add_argument("--run-name", default=None, help="Optional fixed run directory name for resumable runs.")
    return parser


async def run_cli_async(args: argparse.Namespace) -> int:
    preset = _PRESET_DEFAULTS[args.preset]
    max_examples = args.sample_cap if args.sample_cap is not None else preset["max_examples"]
    batch_size = args.batch_size if args.batch_size is not None else preset["batch_size"]
    gpu_utilization = args.gpu_utilization
    if gpu_utilization is None:
        gpu_utilization = 0.33

    config = P2RunConfig(
        output_root=Path(args.output_root),
        dataset_name=args.dataset,
        dataset_alias=args.dataset_alias,
        split=args.split or "validation",
        max_examples=max_examples,
        batch_size=batch_size,
        max_concurrency=args.max_concurrency or preset["max_concurrency"],
        synth_max_concurrency=args.max_concurrency or preset["max_concurrency"],
        run_name=args.run_name,
        provider=args.provider or "openai-compatible",
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        local_launch_port=args.local_launch_port,
        gpu_utilization=gpu_utilization,
    )

    result = await run_p2_suite(config)
    payload = {
        "run_id": result.run_id,
        "output_dir": str(result.output_dir),
        "manifest_path": str(result.manifest_path),
        "prediction_file": str(result.prediction_file),
        "score_file": str(result.score_file),
        "summary_files": [str(path) for path in result.summary_files],
        "aggregate_summary_path": str(result.aggregate_summary_path),
        "total_examples": result.total_examples,
        "completed_examples": result.completed_examples,
        "failed_examples": result.failed_examples,
        "combined_metric": result.combined_metric,
        "dataset_scores": result.dataset_scores,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run_cli_async(args))
