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

from benchmark_runner.config import P2_DATASET_CONFIGS, default_provider_config, get_dataset_configs, get_preset_spec
from benchmark_runner.suite import run_registered_benchmark_suite

from common import get_current_user

_P2_COUNCIL_MODEL = "p2_council"

# Models for P2 — defaults match the vLLM specialist models used on CARC
_P2_API_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
_P2_API_KEY_ENV = "NVIDIA_API_KEY"
_P2_QA_MODEL = "task-aware-llm-council/gemma-2-9b-it-GPTQ"
_P2_REASONING_MODEL = "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"
_P2_GENERAL_MODEL = "task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2"


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
        default=0.33,
        help="Optional provider endpoint override. Required for `local` and useful for other HTTP-backed providers.",
    )

    parser.add_argument(
        "--api-key-env",
        help="Optional environment variable name for provider auth. Usually not needed for local vLLM/OpenAI-compatible servers.",
    )
    parser.add_argument(
        "--policy",
        choices=("p1", "p2"),
        default="p1",
        help="Which council policy to run. p1=single model baseline, p2=flat council with voting and synthesis.",
    )
    return parser


async def run_cli_async(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    spec = get_preset_spec(args.preset, output_root=output_root)
    if args.provider or args.api_base or args.api_key_env:
        provider = args.provider or spec.provider_config.provider

        new_params = dict(spec.provider_config.default_params)
        if provider == "vllm":
            provider = Provider.LOCAL
            new_params[LOCAL_LAUNCH_IMAGE] = "vllm-openai_latest.sif"
            new_params[LOCAL_LAUNCH_BIND] = f"/scratch1/{get_current_user()}/.cache"
            new_params[LOCAL_LAUNCH_STARTUP_TIMEOUT] = 600

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

    if args.policy == "p2":
        result = await _run_p2(spec, args)
    else:
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


async def _run_p2(spec, args):
    from dataclasses import replace as dc_replace
    from model_orchestration import ModelOrchestrator, build_default_orchestrator_config, build_default_local_vllm_orchestrator_config
    from model_orchestration.models import LocalVLLMPresetConfig
    from council_policies.adapter import P2PolicyClient
    from benchmarking_pipeline import run_benchmark

    provider = args.provider or "openai-compatible"

    if provider == "vllm":
        # Local vLLM on CARC: start 3 servers at base_port, base_port+1, base_port+2
        orchestrator_config = build_default_local_vllm_orchestrator_config(
            qa_model=_P2_QA_MODEL,
            reasoning_model=_P2_REASONING_MODEL,
            general_model=_P2_GENERAL_MODEL,
            preset=LocalVLLMPresetConfig(
                base_port=args.local_launch_port,
                gpu_memory_utilization=args.gpu_utilization,
                bind=f"/scratch1/{get_current_user()}/.cache",
            ),
        )
    else:
        # API-based (NVIDIA, OpenRouter, etc.)
        orchestrator_config = build_default_orchestrator_config(
            provider=provider,
            api_base=args.api_base or _P2_API_BASE,
            api_key_env=args.api_key_env or _P2_API_KEY_ENV,
            qa_model=_P2_QA_MODEL,
            reasoning_model=_P2_REASONING_MODEL,
            general_model=_P2_GENERAL_MODEL,
            timeout_seconds=60,
            max_retries=1,
        )

    spec = dc_replace(spec, models=(_P2_COUNCIL_MODEL,))

    async with ModelOrchestrator(orchestrator_config) as orch:
        client = P2PolicyClient(orch, model_name=_P2_COUNCIL_MODEL)

        async def pipeline_runner(datasets, pipeline_config):
            return await run_benchmark(datasets, pipeline_config, client=client)

        return await run_registered_benchmark_suite(
            P2_DATASET_CONFIGS, spec, pipeline_runner=pipeline_runner
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run_cli_async(args))
