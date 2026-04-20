"""
CLI entry point for the P2 benchmark runner.

Usage:
    uv run council-p2-bench

Environment variables:
    OPENROUTER_API_KEY   — for OpenRouter provider
    OPENAI_API_KEY       — for OpenAI provider
    HUGGINGFACE_API_KEY  — for HuggingFace provider

Optional CLI args:
    --provider       openrouter | openai | huggingface   (default: openrouter)
    --qa-model       model name for qa role
    --reasoning-model model name for reasoning role
    --general-model  model name for general role
    --datasets       dataset names to run (default: all 5)
    --n-per-dataset  questions per dataset               (default: 50)
    --output-root    directory to save results           (default: ./p2_benchmark_results)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from llm_gateway import Provider
from model_orchestration import ModelOrchestrator, build_default_orchestrator_config, build_default_local_vllm_orchestrator_config
from model_orchestration.models import LocalVLLMPresetConfig
from task_eval import get_dataset_profile

from council_policies.p2_benchmark import run_p2_benchmark_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"

_DEFAULT_MODELS = {
    "openrouter": {
        "qa": "meta-llama/llama-3.1-8b-instruct",
        "reasoning": "deepseek/deepseek-r1",
        "general": "qwen/qwen-2.5-72b-instruct",
    },
    "openai": {
        "qa": "gpt-4o-mini",
        "reasoning": "o1-mini",
        "general": "gpt-4o",
    },
    "huggingface": {
        "qa": "meta-llama/Llama-3.1-8B-Instruct",
        "reasoning": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "general": "Qwen/Qwen2.5-72B-Instruct",
    },
    "nvidia": {
        "qa": "meta/llama-3.1-8b-instruct",
        "reasoning": "nvidia/llama-3.1-nemotron-70b-instruct",
        "general": "meta/llama-3.3-70b-instruct",
    },
}

_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
}

_PROVIDER_ENUM = {
    "openrouter": Provider.OPENROUTER,
    "openai": Provider.OPENAI,
    "huggingface": Provider.HUGGINGFACE,
    "nvidia": Provider.OPENAI_COMPATIBLE,
    "local": Provider.LOCAL,
}

_ALL_DATASETS = ("musique", "quality", "fever", "hardmath", "humaneval_plus")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P2 council benchmark (same datasets as P1)")
    parser.add_argument("--provider", default="openrouter", choices=list(_PROVIDER_ENUM))
    parser.add_argument("--api-base", default=None, help="Override API base URL (auto-set for nvidia)")
    parser.add_argument("--base-port", type=int, default=8000, help="Base port for local vllm (roles use base, base+1, base+2)")
    parser.add_argument("--qa-model", default=None)
    parser.add_argument("--reasoning-model", default=None)
    parser.add_argument("--general-model", default=None)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset names. Defaults to all 5.")
    parser.add_argument("--n-per-dataset", type=int, default=50)
    parser.add_argument("--output-root", default="p2_benchmark_results",
                        help="Directory to save benchmark output")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    provider_name = args.provider
    provider = _PROVIDER_ENUM[provider_name]

    if provider_name == "local":
        from model_orchestration.defaults import vLLM_DEFAULT_QA_MODEL, vLLM_DEFAULT_REASONING_MODEL, vLLM_DEFAULT_GENERAL_MODEL
        config = build_default_local_vllm_orchestrator_config(
            qa_model=args.qa_model or vLLM_DEFAULT_QA_MODEL,
            reasoning_model=args.reasoning_model or vLLM_DEFAULT_REASONING_MODEL,
            general_model=args.general_model or vLLM_DEFAULT_GENERAL_MODEL,
            preset=LocalVLLMPresetConfig(base_port=args.base_port),
        )
    else:
        defaults = _DEFAULT_MODELS[provider_name]
        api_base = args.api_base or (_NVIDIA_API_BASE if provider_name == "nvidia" else None)
        config = build_default_orchestrator_config(
            provider=provider,
            api_base=api_base,
            api_key_env=_API_KEY_ENV[provider_name],
            qa_model=args.qa_model or defaults["qa"],
            reasoning_model=args.reasoning_model or defaults["reasoning"],
            general_model=args.general_model or defaults["general"],
        )

    dataset_names = args.datasets or list(_ALL_DATASETS)

    # Load dataset profiles — same ones P1 uses
    sources = []
    for name in dataset_names:
        try:
            profile = get_dataset_profile(name)
            sources.append(profile)
            logger.info("Loaded dataset profile: %s", name)
        except Exception as exc:
            logger.warning("Could not load dataset %r: %s — skipping.", name, exc)

    if not sources:
        logger.error("No datasets loaded. Exiting.")
        return

    async with ModelOrchestrator(config) as orchestrator:
        result = await run_p2_benchmark_suite(
            sources,
            orchestrator,
            output_root=Path(args.output_root),
            n_per_dataset=args.n_per_dataset,
            metric_resolver=lambda source: source,
        )

    print(f"\n=== P2 Benchmark Complete ===")
    print(f"Suite ID   : {result.suite_id}")
    print(f"Output dir : {result.output_dir}")
    print(f"Total      : {result.total_examples} questions")
    print(f"Scored     : {result.scored_examples}")
    print(f"Failed     : {result.failed_examples}")
    print(f"Skipped    : {result.skipped_examples}")
    print(f"\nAggregate summary: {result.aggregate_summary_path}")

    # Print per-dataset council winner
    agg = json.loads(result.aggregate_summary_path.read_text())
    print("\n=== Dataset Council Winners ===")
    for row in agg:
        scores = ", ".join(f"{r}={v:.1f}" for r, v in row.get("scores_by_role", {}).items())
        print(f"  {row['dataset_name']:<20} winner={row['council_winner']:<12} scores=({scores})")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
