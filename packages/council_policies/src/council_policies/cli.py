from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from llm_gateway import Provider, PromptRequest, ProviderConfig
from model_orchestration.defaults import (
    API_DEFAULT_GENERAL_MODEL,
    API_DEFAULT_QA_MODEL,
    API_DEFAULT_REASONING_MODEL,
    build_default_local_vllm_orchestrator_config,
    build_default_orchestrator_config,
    vLLM_DEFAULT_GENERAL_MODEL,
    vLLM_DEFAULT_QA_MODEL,
    vLLM_DEFAULT_REASONING_MODEL,
)
from model_orchestration.models import LocalVLLMPresetConfig, OrchestratorConfig
from benchmarking_pipeline.models import (
    BenchmarkDataset,
    BenchmarkPrediction,
    BenchmarkRunConfig,
    BenchmarkRunResult,
    default_run_id,
)
from benchmarking_pipeline.storage import (
    append_prediction,
    ensure_run_directory,
    load_recorded_example_ids,
    prediction_path,
    write_manifest as write_run_manifest,
)
from benchmark_runner.config import get_dataset_configs
from benchmark_runner.models import BenchmarkSpec, default_suite_id
from benchmark_runner.suite import run_registered_benchmark_suite
from common import get_current_user

from council_policies.policy_adapters import P3Adapter
from council_policies.policy_runner import PolicyRuntime, SpecialistCache, SpecialistRequest

logger = logging.getLogger(__name__)

POLICY_LABEL = "p3"

# Dummy ProviderConfig so run_benchmark_suite won't try to launch a vLLM server
# for model="p3".  P3 manages its own ModelOrchestrator internally.
_DUMMY_PROVIDER = ProviderConfig(
    provider=Provider.OPENAI_COMPATIBLE,
    api_base="http://localhost:0",
)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run P3 rule-based routing policy benchmark across all datasets."
    )
    parser.add_argument(
        "--output-root",
        default=f"/scratch1/{get_current_user()}/results",
        help="Root directory for suite output.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset subset. Defaults to all configured datasets.",
    )
    parser.add_argument(
        "--sample-cap",
        type=int,
        default=500,
        help="Max examples per dataset.",
    )
    parser.add_argument(
        "--split",
        help="Split override applied to all selected datasets.",
    )
    parser.add_argument(
        "--provider",
        choices=("local", "vllm", "openai", "openrouter", "openai-compatible"),
        default="local",
        help="LLM provider backend.",
    )
    parser.add_argument(
        "--api-base",
        help="Provider API base URL. Required for external HTTP providers.",
    )
    parser.add_argument(
        "--local-launch-port",
        type=int,
        default=8000,
        help="Base port for local vLLM. Specialists use base, base+1, base+2.",
    )
    parser.add_argument(
        "--gpu-utilization",
        type=float,
        default=0.33,
        help="GPU memory utilization fraction per local vLLM model.",
    )
    parser.add_argument(
        "--api-key-env",
        help="Env variable name containing the provider API key.",
    )
    parser.add_argument("--qa-model", help="Override QA specialist model name.")
    parser.add_argument("--reasoning-model", help="Override reasoning/math/code specialist model.")
    parser.add_argument("--general-model", help="Override general/FEVER specialist model.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=100,
        help="Max concurrent LLM requests per batch.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _build_orchestrator_config(args: argparse.Namespace) -> OrchestratorConfig:
    provider_str = "local" if args.provider == "vllm" else args.provider
    is_local = (provider_str == "local") and not args.api_base

    if is_local:
        preset = LocalVLLMPresetConfig(
            base_port=args.local_launch_port,
            gpu_memory_utilization=args.gpu_utilization,
        )
        return build_default_local_vllm_orchestrator_config(
            qa_model=args.qa_model or vLLM_DEFAULT_QA_MODEL,
            reasoning_model=args.reasoning_model or vLLM_DEFAULT_REASONING_MODEL,
            general_model=args.general_model or vLLM_DEFAULT_GENERAL_MODEL,
            preset=preset,
        )

    return build_default_orchestrator_config(
        provider=provider_str,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        qa_model=args.qa_model or API_DEFAULT_QA_MODEL,
        reasoning_model=args.reasoning_model or API_DEFAULT_REASONING_MODEL,
        general_model=args.general_model or API_DEFAULT_GENERAL_MODEL,
    )


def _build_spec(args: argparse.Namespace) -> BenchmarkSpec:
    return BenchmarkSpec(
        provider_config=_DUMMY_PROVIDER,
        models=(POLICY_LABEL,),
        output_root=Path(args.output_root),
        suite_name=None,
        batch_size=args.sample_cap,
        max_concurrency=args.max_concurrency,
        max_examples_per_dataset=args.sample_cap,
        continue_on_error=True,
        skip_existing_predictions=True,
        delay_between_requests=0,
    )


# ---------------------------------------------------------------------------
# P3 pipeline runner (plugged into run_benchmark_suite)
# ---------------------------------------------------------------------------

async def _run_p3_pipeline(
    datasets: Iterable[BenchmarkDataset],
    config: BenchmarkRunConfig,
    *,
    policy: P3Adapter,
    runtime: PolicyRuntime,
    max_concurrency: int,
) -> BenchmarkRunResult:
    """Drop-in replacement for run_benchmark that uses the P3 policy."""
    dataset_list = tuple(datasets)
    run_id = config.run_name or default_run_id()
    run_dir = ensure_run_directory(Path(config.output_root), run_id)
    write_run_manifest(run_dir, run_id=run_id, config=config, datasets=dataset_list)

    created_at = datetime.now(timezone.utc).isoformat()
    prediction_files: set[Path] = set()
    total_examples = 0
    attempted_examples = 0
    completed_examples = 0
    failed_examples = 0
    skipped_existing = 0

    model = config.models[0]  # "p3"

    for dataset in dataset_list:
        path = prediction_path(run_dir, dataset.name, model)
        prediction_files.add(path)

        completed_ids = load_recorded_example_ids(path) if config.skip_existing else set()
        examples_to_run = [e for e in dataset.examples if e.example_id not in completed_ids]
        skipped_existing += len(dataset.examples) - len(examples_to_run)

        if not examples_to_run:
            logger.info("Skipping %s — all %d examples already predicted.", dataset.name, len(dataset.examples))
            continue

        total_examples += len(examples_to_run)
        logger.info("Running P3 on %d examples for dataset '%s'.", len(examples_to_run), dataset.name)

        requests = [_build_request(e, model=model, config=config) for e in examples_to_run]
        results = await _execute_p3_batch(policy, runtime, requests, max_concurrency=max_concurrency)

        for example, req, result in zip(examples_to_run, requests, results):
            attempted_examples += 1
            if isinstance(result, Exception):
                pred = BenchmarkPrediction(
                    run_id=run_id,
                    dataset_name=dataset.name,
                    model=model,
                    example_id=example.example_id,
                    status="error",
                    prompt_version=config.prompt_version,
                    example_metadata=dict(example.metadata),
                    request_metadata=dict(req.metadata),
                    error_type=type(result).__name__,
                    error_message=str(result),
                )
                failed_examples += 1
                logger.warning(
                    "Example %s failed: %s: %s",
                    example.example_id, type(result).__name__, result,
                )
            else:
                resp = result.response
                pred = BenchmarkPrediction(
                    run_id=run_id,
                    dataset_name=dataset.name,
                    model=model,
                    example_id=example.example_id,
                    status="success",
                    prompt_version=config.prompt_version,
                    response_text=resp.text,
                    latency_ms=resp.latency_ms,
                    request_id=resp.request_id,
                    finish_reason=resp.finish_reason,
                    provider=resp.provider,
                    usage=asdict(resp.usage),
                    example_metadata=dict(example.metadata),
                    request_metadata=dict(req.metadata),
                    response_metadata=dict(resp.metadata),
                )
                completed_examples += 1
            append_prediction(path, pred)

        logger.info(
            "Dataset '%s' done: %d succeeded, %d failed.",
            dataset.name,
            completed_examples,
            failed_examples,
        )

    return BenchmarkRunResult(
        run_id=run_id,
        output_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        created_at=created_at,
        total_examples=total_examples,
        attempted_examples=attempted_examples,
        completed_examples=completed_examples,
        failed_examples=failed_examples,
        skipped_existing=skipped_existing,
        prediction_files=tuple(sorted(prediction_files)),
    )


def _build_request(
    example,
    *,
    model: str,
    config: BenchmarkRunConfig,
) -> PromptRequest:
    metadata: dict = dict(example.metadata)
    metadata.update(
        {
            "dataset_name": example.dataset_name,
            "example_id": example.example_id,
            "prompt_version": config.prompt_version,
        }
    )
    if example.messages:
        return PromptRequest(
            model=model,
            messages=example.messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop_sequences=config.stop_sequences,
            metadata=metadata,
            provider_params=dict(config.provider_params),
        )
    return PromptRequest(
        model=model,
        system_prompt=example.system_prompt,
        user_prompt=example.question,
        context=example.context,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stop_sequences=config.stop_sequences,
        metadata=metadata,
        provider_params=dict(config.provider_params),
    )


async def _execute_p3_batch(
    policy: P3Adapter,
    runtime: PolicyRuntime,
    requests: list[PromptRequest],
    *,
    max_concurrency: int,
) -> list[object]:
    """
    Plan, execute, and finalize one batch of requests through P3.

    Does NOT close the runtime — callers are responsible for lifecycle.
    Returns a list parallel to `requests`: each entry is either a
    PolicyResult (success) or an Exception (failure).
    """
    cache = SpecialistCache()

    states = []
    for req in requests:
        state = await policy.plan(req, runtime)
        states.append(state)

    all_srs: list[SpecialistRequest] = [
        sr for state in states for sr in state.specialist_requests
    ]
    seen: set[str] = set()
    unique_srs: list[SpecialistRequest] = []
    for sr in all_srs:
        if sr.cache_key not in seen:
            seen.add(sr.cache_key)
            unique_srs.append(sr)

    error_by_key: dict[str, Exception] = {}
    semaphore = asyncio.Semaphore(max_concurrency)

    await runtime.open_specialists()

    async def _run_one(sr: SpecialistRequest) -> None:
        async with semaphore:
            try:
                response = await runtime.specialist_orchestrator.get_client(
                    sr.role
                ).get_response(sr.request)
                cache.put(sr.cache_key, response)
            except Exception as exc:
                error_by_key[sr.cache_key] = exc

    await asyncio.gather(*(_run_one(sr) for sr in unique_srs))

    results: list[object] = []
    for state in states:
        sr = state.specialist_requests[0]
        if sr.cache_key in error_by_key:
            results.append(error_by_key[sr.cache_key])
        else:
            updated_state = await policy.complete_specialist_phase(state, cache, runtime)
            result = await policy.finalize(updated_state, cache, runtime)
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_cli_async(args: argparse.Namespace) -> int:
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    orchestrator_config = _build_orchestrator_config(args)
    spec = _build_spec(args)

    policy = P3Adapter(fallback_role="general")
    runtime = PolicyRuntime(
        specialist_config=orchestrator_config,
        synthesizer_config=orchestrator_config,
    )

    dataset_configs = get_dataset_configs(tuple(args.datasets) if args.datasets else None)
    if args.split:
        dataset_configs = tuple(
            type(dc)(name=dc.name, split=args.split, profile_kwargs=dict(dc.profile_kwargs))
            for dc in dataset_configs
        )

    max_concurrency = args.max_concurrency

    async def _pipeline_runner(datasets, config):
        return await _run_p3_pipeline(
            datasets,
            config,
            policy=policy,
            runtime=runtime,
            max_concurrency=max_concurrency,
        )

    async with runtime:
        result = await run_registered_benchmark_suite(
            dataset_configs,
            spec,
            pipeline_runner=_pipeline_runner,
        )

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
