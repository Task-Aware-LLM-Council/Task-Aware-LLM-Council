"""
P1 benchmark CLI — runs individual models on the same datasets as P2 and scores
with task-eval metrics. Produces JSONL output comparable to council-p2-bench.

Usage:
    uv run council-p1 --provider nvidia --n-per-dataset 2 --output-root ./p1_bench_output

Environment variables:
    NVIDIA_API_KEY        — for nvidia provider
    OPENROUTER_API_KEY    — for openrouter provider
    OPENAI_API_KEY        — for openai provider
    HUGGINGFACE_API_KEY   — for huggingface provider
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_gateway import Provider, PromptRequest
from model_orchestration import ModelOrchestrator, build_default_orchestrator_config
from task_eval import get_dataset_profile
from task_eval.models import EvaluationCase

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
}

_ALL_DATASETS = ("musique", "quality", "fever", "hardmath", "humaneval_plus")


# ── Output models ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class P1ScoreRecord:
    run_id: str
    dataset_name: str
    example_id: str
    role: str
    model: str
    status: str                   # "scored" | "error" | "empty"
    response_text: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


@dataclass(frozen=True)
class P1DatasetSummary:
    run_id: str
    dataset_name: str
    role: str
    model: str
    total_examples: int = 0
    scored_examples: int = 0
    failed_examples: int = 0
    aggregated_metrics: dict[str, float] = field(default_factory=dict)


# ── Storage ────────────────────────────────────────────────────────────────────

def _append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


# ── Core runner ────────────────────────────────────────────────────────────────

async def run_p1_benchmark(
    orchestrator: ModelOrchestrator,
    *,
    roles: list[str],
    dataset_names: list[str],
    n_per_dataset: int,
    output_root: Path,
    run_id: str,
) -> dict[str, Any]:
    suite_dir = output_root / run_id
    (suite_dir / "scores").mkdir(parents=True, exist_ok=True)
    (suite_dir / "summaries").mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []
    total = scored = failed = 0

    for dataset_name in dataset_names:
        try:
            profile = get_dataset_profile(dataset_name)
        except Exception as exc:
            logger.warning("Could not load dataset %r: %s — skipping.", dataset_name, exc)
            continue

        # Load cases
        cases: list[EvaluationCase] = []
        for case in profile.iter_cases():
            cases.append(case)
            if len(cases) >= n_per_dataset:
                break

        if not cases:
            logger.warning("No cases for %r — skipping.", dataset_name)
            continue

        logger.info("P1 benchmark: %d cases from %r", len(cases), dataset_name)

        for role in roles:
            score_file = suite_dir / "scores" / f"{dataset_name}__{role}.jsonl"
            records: list[dict[str, Any]] = []

            # Get model name for this role
            try:
                client = orchestrator.get_client(role)
                model_name = getattr(client, "model", role)
            except KeyError:
                logger.warning("Role %r not registered — skipping.", role)
                continue

            async def _run_one(case: EvaluationCase) -> P1ScoreRecord:
                example = case.example
                request = PromptRequest(
                    user_prompt=example.question,
                    context=example.context,
                    system_prompt=example.system_prompt,
                    messages=example.messages,
                )
                try:
                    response = await orchestrator.get_client(role).get_response(request)
                    text = response.text.strip()
                    if not text:
                        return P1ScoreRecord(
                            run_id=run_id,
                            dataset_name=dataset_name,
                            example_id=example.example_id,
                            role=role,
                            model=model_name,
                            status="empty",
                            reference=dict(case.reference),
                        )
                    prediction = {"response_text": text, "status": "success"}
                    metric_result = profile.score(
                        case=case,
                        prediction=prediction,
                        dataset_metadata={},
                    )
                    return P1ScoreRecord(
                        run_id=run_id,
                        dataset_name=dataset_name,
                        example_id=example.example_id,
                        role=role,
                        model=model_name,
                        status="scored",
                        response_text=text,
                        metrics=dict(metric_result.values),
                        reference=dict(case.reference),
                    )
                except Exception as exc:
                    logger.warning("Role %r failed on %s: %s", role, example.example_id, exc)
                    return P1ScoreRecord(
                        run_id=run_id,
                        dataset_name=dataset_name,
                        example_id=example.example_id,
                        role=role,
                        model=model_name,
                        status="error",
                        reference=dict(case.reference),
                        error_message=str(exc),
                    )

            results = await asyncio.gather(*[_run_one(c) for c in cases])

            for record in results:
                _append_jsonl(score_file, record)
                records.append(asdict(record))
                total += 1
                if record.status == "scored":
                    scored += 1
                else:
                    failed += 1

            # Aggregate metrics
            scored_records = [r for r in records if r["status"] == "scored"]
            agg: dict[str, float] = {}
            if scored_records:
                from task_eval.scoring import aggregate_numeric_metrics
                agg = aggregate_numeric_metrics(scored_records)

            summary = P1DatasetSummary(
                run_id=run_id,
                dataset_name=dataset_name,
                role=role,
                model=model_name,
                total_examples=len(records),
                scored_examples=len(scored_records),
                failed_examples=len(records) - len(scored_records),
                aggregated_metrics=agg,
            )
            summary_file = suite_dir / "summaries" / f"{dataset_name}__{role}.json"
            _write_json(summary_file, asdict(summary))

            aggregate_rows.append({
                "run_id": run_id,
                "dataset_name": dataset_name,
                "role": role,
                "model": model_name,
                "total_examples": summary.total_examples,
                "scored_examples": summary.scored_examples,
                "aggregated_metrics": agg,
            })

            logger.info(
                "  %-20s role=%-12s scored=%d/%d metrics=%s",
                dataset_name, role, summary.scored_examples, summary.total_examples,
                {k: f"{v:.3f}" for k, v in agg.items()},
            )

    agg_path = suite_dir / "suite_metrics.json"
    _write_json(agg_path, aggregate_rows)
    logger.info("P1 benchmark complete. Results saved to %s", suite_dir)
    return {"run_id": run_id, "output_dir": str(suite_dir), "total": total, "scored": scored, "failed": failed}


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P1 individual model benchmark")
    parser.add_argument("--provider", default="openrouter", choices=list(_PROVIDER_ENUM))
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--qa-model", default=None)
    parser.add_argument("--reasoning-model", default=None)
    parser.add_argument("--general-model", default=None)
    parser.add_argument("--roles", nargs="*", default=["qa", "general"],
                        help="Which roles to benchmark (default: qa general)")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--n-per-dataset", type=int, default=50)
    parser.add_argument("--output-root", default="p1_bench_output")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    provider_name = args.provider
    defaults = _DEFAULT_MODELS[provider_name]
    provider = _PROVIDER_ENUM[provider_name]
    api_base = args.api_base or (_NVIDIA_API_BASE if provider_name == "nvidia" else None)

    config = build_default_orchestrator_config(
        provider=provider,
        api_base=api_base,
        api_key_env=_API_KEY_ENV[provider_name],
        qa_model=args.qa_model or defaults["qa"],
        reasoning_model=args.reasoning_model or defaults["reasoning"],
        general_model=args.general_model or defaults["general"],
    )

    run_id = datetime.now(timezone.utc).strftime("p1_run_%Y%m%dT%H%M%SZ")
    dataset_names = args.datasets or list(_ALL_DATASETS)

    async with ModelOrchestrator(config) as orchestrator:
        summary = await run_p1_benchmark(
            orchestrator,
            roles=args.roles,
            dataset_names=dataset_names,
            n_per_dataset=args.n_per_dataset,
            output_root=Path(args.output_root),
            run_id=run_id,
        )

    print(f"\n=== P1 Benchmark Complete ===")
    print(f"Run ID    : {summary['run_id']}")
    print(f"Output    : {summary['output_dir']}")
    print(f"Total     : {summary['total']}")
    print(f"Scored    : {summary['scored']}")
    print(f"Failed    : {summary['failed']}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
