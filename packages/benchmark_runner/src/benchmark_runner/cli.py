from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from llm_gateway import (
    LOCAL_LAUNCH_BIND,
    LOCAL_LAUNCH_DTYPE,
    LOCAL_LAUNCH_PORT,
    LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION,
    LOCAL_LAUNCH_IMAGE,
    LOCAL_LAUNCH_LOAD_FORMAT,
    LOCAL_LAUNCH_MAX_MODEL_LEN,
    LOCAL_LAUNCH_QUANTIZATION,
    LOCAL_LAUNCH_STARTUP_TIMEOUT,
    Provider,
)

from benchmark_runner.config import default_provider_config, get_dataset_configs, get_preset_spec
from benchmark_runner.suite import run_registered_benchmark_suite

from common import get_current_user


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark model-dataset pairs.")
    parser.add_argument("--preset", default="pilot", choices=("pilot", "full"))
    parser.add_argument(
        "--output-root",
        default=f"/scratch1/{get_current_user()}/results/benchmark_suite",
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
    parser.add_argument(
        "--provider",
        choices=("local", "vllm", "huggingface", "openai", "openrouter", "openai-compatible"),
        help="Optional provider override. Use `local` or `vllm` for an external local OpenAI-compatible server.",
    )
    parser.add_argument(
        "--api-base",
        help="Optional provider endpoint override. Required for `local` and useful for other HTTP-backed providers.",
    )

    parser.add_argument(
        "--local-launch-port",
        type=int,
        default=8000,
        help="Optional provider endpoint override. Required for `local` and useful for other HTTP-backed providers.",
    )

    parser.add_argument(
        "--gpu-utilization",
        type=float,
        default=None,
        help=(
            "GPU memory utilization fraction for local vLLM servers. "
            "Defaults to 0.90 with --single-gpu, 0.33 otherwise."
        ),
    )

    parser.add_argument(
        "--api-key-env",
        help="Optional environment variable name for provider auth. Usually not needed for local vLLM/OpenAI-compatible servers.",
    )
    parser.add_argument(
        "--suite-name",
        default=None,
        help=(
            "Fixed suite directory name (e.g. 'suite_20260421T170856Z'). "
            "Re-using an existing name with skip_existing=True lets you resume a partial run "
            "without re-running already-completed dataset/model pairs."
        ),
    )
    return parser


async def run_cli_async(args: argparse.Namespace) -> int:
    if args.gpu_utilization is None:
        args.gpu_utilization = 0.33

    output_root = Path(args.output_root)
    spec = get_preset_spec(args.preset, output_root=output_root)
    if args.suite_name:
        spec = replace(spec, suite_name=args.suite_name)
    if args.provider or args.api_base or args.api_key_env:
        provider = args.provider or spec.provider_config.provider

        new_params = dict(spec.provider_config.default_params)
        if provider == "vllm":
            provider = Provider.LOCAL
            new_params[LOCAL_LAUNCH_IMAGE] = f"/scratch1/{get_current_user()}/Task-Aware-LLM-Council/vllm-openai_latest.sif"
            new_params[LOCAL_LAUNCH_BIND] = f"/scratch1/{get_current_user()}/.cache"
            new_params[LOCAL_LAUNCH_STARTUP_TIMEOUT] = 1200

            # new_params[LOCAL_LAUNCH_QUANTIZATION] = 'bitsandbytes'
            # new_params[LOCAL_LAUNCH_LOAD_FORMAT] = 'bitsandbytes'
            # new_params[LOCAL_LAUNCH_DTYPE] = 'bfloat16'
            new_params[LOCAL_LAUNCH_MAX_MODEL_LEN] = '8192'
            new_params[LOCAL_LAUNCH_GPU_MEMORY_UTILIZATION] = args.gpu_utilization
            new_params[LOCAL_LAUNCH_PORT] = args.local_launch_port
            new_params[LOCAL_LAUNCH_QUANTIZATION] = 'compressed-tensors'
        
        base_provider_config = default_provider_config(
            provider=provider,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
        )
        configured_provider = replace(base_provider_config, default_params=new_params)
        spec = replace(
            spec,
            provider_config=configured_provider
        )

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
