from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator
import ast

from datasets import load_dataset
from llm_gateway import PromptRequest, Provider, ProviderConfig
from llm_gateway.models import Usage
from model_orchestration import (
    DEFAULT_LOCAL_VLLM_BIND,
    LocalVLLMPresetConfig,
    ModelOrchestrator,
    ModelSpec,
    OrchestratorConfig,
    build_default_local_vllm_orchestrator_config,
    build_default_orchestrator_config,
)
from task_eval.extraction import (
    extract_code_answer,
    extract_fever_label,
    extract_math_answer,
    extract_qa_answer,
    extract_qa_answer_musique,
)
from task_eval.scoring import exact_match_multi, label_accuracy, math_exact_match, pass_at_1, token_f1_multi

from council_policies.adapter import P2PolicyClient
from council_policies.models import P2CouncilDecision, P2RunConfig, P2RunResult, P2ScoreRecord, P2SynthesizedRecord
from council_policies.p2.prompts import build_specialist_prompt, build_synthesis_prompt

P2_COUNCIL_MODEL = "p2_council"
P2_DEFAULT_DATASET = "task-aware-llm-council/router_dataset-2"
P2_DEFAULT_DATASET_ALIAS = "router_dataset"

P2_API_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
P2_API_KEY_ENV = "NVIDIA_API_KEY"
P2_API_QA_MODEL = "moonshotai/kimi-k2-instruct-0905"
P2_API_REASONING_MODEL = "moonshotai/kimi-k2-instruct-0905"
P2_API_GENERAL_MODEL = "moonshotai/kimi-k2-instruct-0905"
P2_API_SYNTH_MODEL = "moonshotai/kimi-k2-instruct-0905"
P2_VLLM_SYNTH_MODEL = "task-aware-llm-council/Qwen2.5-7B-Instruct-AWQ"

P2_MANIFEST_FILENAME = "manifest.json"
P2_AGGREGATE_SUMMARY_FILENAME = "suite_metrics.json"
P2_PROMPT_VERSION = "p2"
_MUSIQUE_NOT_PRESENT = "NOT PRESENT IN CONTEXT"
_DATASET_ORDER = ("musique", "quality", "fever", "hardmath", "humaneval_plus")


async def run_p2_suite(config: P2RunConfig) -> P2RunResult:
    run_id = config.run_name or _default_run_id()
    output_dir = Path(config.output_root) / run_id
    prediction_file = _prediction_path(output_dir, config.dataset_alias, P2_COUNCIL_MODEL)
    score_file = _score_path(output_dir, config.dataset_alias, P2_COUNCIL_MODEL)
    aggregate_summary_path = output_dir / P2_AGGREGATE_SUMMARY_FILENAME
    manifest_path = output_dir / P2_MANIFEST_FILENAME

    phase_status = {"specialist": "pending", "synthesizer": "pending", "scoring": "pending"}
    _reset_jsonl_file(prediction_file)
    _reset_jsonl_file(score_file)

    summary_state = _new_summary_state()
    _write_manifest(
        manifest_path,
        config=config,
        run_id=run_id,
        total_examples=0,
        phase_status=phase_status,
        processed_examples=0,
        completed_examples=0,
        failed_examples=0,
        current_batch_index=0,
    )
    row_iter = _iter_dataset_rows(config)
    batch_index = 0
    row_offset = 0
    total_examples = 0
    completed_examples = 0
    failed_examples = 0
    specialist_records: list[dict[str, Any]] = []
    decision_payloads: list[dict[str, Any]] = []

    print("Running specialist phase")
    async with ModelOrchestrator(_build_specialist_orchestrator_config(config)) as specialist_orch:
        await specialist_orch.load_all(max_parallel=1)
        specialist_client = P2PolicyClient(specialist_orch, model_name=P2_COUNCIL_MODEL)

        for batch_index, rows in enumerate(_batched_rows(row_iter, max(config.batch_size, 1)), start=1):
            total_examples += len(rows)
            batch_records, batch_payloads = await _run_specialist_batch(
                specialist_client,
                config,
                rows,
                start_index=row_offset,
            )
            specialist_records.extend(batch_records)
            decision_payloads.extend(batch_payloads)
            row_offset += len(rows)

            _write_manifest(
                manifest_path,
                config=config,
                run_id=run_id,
                total_examples=total_examples,
                phase_status=phase_status,
                processed_examples=total_examples,
                completed_examples=0,
                failed_examples=0,
                current_batch_index=batch_index,
            )
    print("specialist phase done")
    phase_status["specialist"] = "completed"
    _write_manifest(
        manifest_path,
        config=config,
        run_id=run_id,
        total_examples=total_examples,
        phase_status=phase_status,
        processed_examples=total_examples,
        completed_examples=0,
        failed_examples=0,
        current_batch_index=batch_index,
    )

    print("Running synthesizer phase")
    synthesized_records: list[dict[str, Any]] = []
    synth_offset = 0
    async with ModelOrchestrator(_build_synthesizer_orchestrator_config(config)) as synth_orch:
        await synth_orch.load_all(max_parallel=1)
        for synth_batch in _batched_rows(decision_payloads, max(config.batch_size, 1)):
            synthesized_records.extend(
                await _run_synthesizer_batch(
                    synth_orch,
                    config,
                    synth_batch,
                )
            )
            synth_offset += len(synth_batch)
            _write_manifest(
                manifest_path,
                config=config,
                run_id=run_id,
                total_examples=total_examples,
                phase_status=phase_status,
                processed_examples=total_examples,
                completed_examples=0,
                failed_examples=0,
                current_batch_index=synth_offset,
            )
    print("synthesizer phase done")
    phase_status["synthesizer"] = "completed"

    score_records, dataset_scores, combined_metric = _score_synthesized_records(
        synthesized_records,
        specialist_records=specialist_records,
        run_id=run_id,
        dataset_name=config.dataset_alias,
    )
    score_records.extend(_build_specialist_error_score_records(run_id, config.dataset_alias, specialist_records))
    phase_status["scoring"] = "completed"

    prediction_records = _build_prediction_records(
        run_id=run_id,
        config=config,
        specialist_records=specialist_records,
        synthesized_records=synthesized_records,
        score_records=score_records,
    )

    _append_jsonl(prediction_file, prediction_records)
    _append_jsonl(score_file, score_records)
    _accumulate_summary_state(summary_state, prediction_records, score_records)

    completed_examples = sum(1 for row in score_records if row.get("status") == "scored")
    failed_examples = total_examples - completed_examples

    dataset_scores, combined_metric = _compute_dataset_scores(summary_state)
    summary_files, aggregate_rows = _write_summaries(
        output_dir=output_dir,
        run_id=run_id,
        dataset_name=config.dataset_alias,
        summary_state=summary_state,
        dataset_scores=dataset_scores,
        combined_metric=combined_metric,
        phase_status=phase_status,
        total_examples=total_examples,
    )
    _write_json(aggregate_summary_path, aggregate_rows)

    result = P2RunResult(
        run_id=run_id,
        output_dir=output_dir,
        manifest_path=manifest_path,
        prediction_file=prediction_file,
        score_file=score_file,
        summary_files=tuple(summary_files),
        aggregate_summary_path=aggregate_summary_path,
        total_examples=total_examples,
        completed_examples=completed_examples,
        failed_examples=failed_examples,
        combined_metric=combined_metric,
        dataset_scores=dataset_scores,
    )
    _write_manifest(
        manifest_path,
        config=config,
        run_id=run_id,
        total_examples=total_examples,
        phase_status=phase_status,
        processed_examples=total_examples,
        completed_examples=completed_examples,
        failed_examples=failed_examples,
        current_batch_index=batch_index,
        result=result,
    )
    return result


async def _run_specialist_batch(
    client: P2PolicyClient,
    config: P2RunConfig,
    rows: list[dict[str, Any]],
    *,
    start_index: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("config.max_concurrency:", config.max_concurrency)
    sem = asyncio.Semaphore(config.max_concurrency)
    specialist_records: list[dict[str, Any]] = []
    decision_payloads: list[dict[str, Any]] = []

    async def process_row(index: int, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        async with sem:
            example_id, request, row_metadata = _build_request_for_row(config, start_index + index, row)
            try:
                decision = await client.generate_decision(
                    request,
                    example_id=example_id,
                    dataset_name=config.dataset_alias,
                )
                record = _serialize_decision(decision, row_metadata=row_metadata)
                return record, {"decision": decision, "row_metadata": row_metadata}
            except Exception as exc:
                return {
                    "example_id": example_id,
                    "dataset_name": config.dataset_alias,
                    "status": "error",
                    "request": _to_jsonable(request),
                    "row_metadata": row_metadata,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }, None

    results = await asyncio.gather(*[process_row(index, row) for index, row in enumerate(rows)])

    for record, payload in results:
        specialist_records.append(record)
        if payload is not None:
            decision_payloads.append(payload)
    return specialist_records, decision_payloads


async def _run_synthesizer_batch(
    orch: ModelOrchestrator,
    config: P2RunConfig,
    decision_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    print("config.max_concurrency:", config.max_concurrency)
    if not decision_payloads:
        return []

    sem = asyncio.Semaphore(config.synth_max_concurrency or config.max_concurrency)

    async def synthesize_one(payload: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            decision: P2CouncilDecision = payload["decision"]
            row_metadata: dict[str, Any] = payload["row_metadata"]
            other_answers = [
                result.text
                for role, result in decision.role_results.items()
                if role != decision.winning_role and result.text
            ]
            synth_request = PromptRequest(
                user_prompt=build_synthesis_prompt(
                    source_dataset=row_metadata["source_dataset"],
                    question=str(row_metadata.get("question") or ""),
                    context=decision.request.context,
                    winning_answer=decision.winning_answer,
                    other_answers=other_answers,
                ),
                temperature=0.0,
            )
            try:
                response = await orch.run(synth_request, target="synthesizer")
                prompt_response = getattr(response, "prompt_response", None)
                synth_record = P2SynthesizedRecord(
                    example_id=decision.example_id,
                    dataset_name=decision.dataset_name,
                    source_dataset=_canonical_source_dataset(row_metadata["source_dataset"]),
                    synthesized_text=response.text.strip(),
                    metric_name=_metric_name_for_row_metadata(row_metadata),
                    request=synth_request,
                    winning_answer=decision.winning_answer,
                    winning_role=decision.winning_role,
                    winning_model=decision.winning_model,
                    metadata={
                        "row_metadata": row_metadata,
                        "specialist_metadata": decision.metadata,
                    },
                    usage=_response_usage(response),
                    latency_ms=getattr(response, "latency_ms", None),
                )
                return {
                    "example_id": decision.example_id,
                    "dataset_name": decision.dataset_name,
                    "source_dataset": synth_record.source_dataset,
                    "status": "success",
                    "metric_name": synth_record.metric_name,
                    "synthesized_text": synth_record.synthesized_text,
                    "request": _to_jsonable(synth_record.request),
                    "winning_answer": synth_record.winning_answer,
                    "winning_role": synth_record.winning_role,
                    "winning_model": synth_record.winning_model,
                    "usage": _to_jsonable(synth_record.usage),
                    "latency_ms": synth_record.latency_ms,
                    "provider": getattr(response, "provider", None),
                    "model": getattr(response, "model", None) or (config.synth_model or P2_API_SYNTH_MODEL),
                    "request_id": getattr(prompt_response, "request_id", None),
                    "finish_reason": getattr(prompt_response, "finish_reason", None),
                    "metadata": _to_jsonable(synth_record.metadata),
                }
            except Exception as exc:
                return {
                    "example_id": decision.example_id,
                    "dataset_name": decision.dataset_name,
                    "source_dataset": _canonical_source_dataset(row_metadata["source_dataset"]),
                    "status": "error",
                    "metric_name": _metric_name_for_row_metadata(row_metadata),
                    "request": _to_jsonable(synth_request),
                    "winning_answer": decision.winning_answer,
                    "winning_role": decision.winning_role,
                    "winning_model": decision.winning_model,
                    "latency_ms": None,
                    "usage": _to_jsonable(Usage()),
                    "model": config.synth_model or P2_API_SYNTH_MODEL,
                    "metadata": {"row_metadata": row_metadata, "specialist_metadata": decision.metadata},
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }

    return list(await asyncio.gather(*[synthesize_one(payload) for payload in decision_payloads]))


def _score_synthesized_records(
    synthesized_records: list[dict[str, Any]],
    *,
    specialist_records: list[dict[str, Any]] | None = None,
    run_id: str = "p2_run",
    dataset_name: str = P2_DEFAULT_DATASET_ALIAS,
) -> tuple[list[dict[str, Any]], dict[str, float], float | None]:
    score_records: list[dict[str, Any]] = []
    per_dataset_primary: dict[str, list[float]] = {key: [] for key in _DATASET_ORDER}
    specialist_by_id = {str(record.get("example_id")): record for record in (specialist_records or [])}

    for record in synthesized_records:
        example_id = str(record.get("example_id"))
        specialist = specialist_by_id.get(example_id, {})
        row_metadata = dict((record.get("metadata") or {}).get("row_metadata") or specialist.get("row_metadata") or {})
        source_key = _canonical_source_dataset(row_metadata.get("source_dataset", record.get("source_dataset", "")))
        row_metrics = _build_row_system_metrics(specialist, record)

        if record.get("status") != "success":
            score_records.append({
                "suite_id": run_id,
                "dataset_name": dataset_name,
                "model": P2_COUNCIL_MODEL,
                "example_id": example_id,
                "status": "error",
                "prediction_status": "error",
                "metrics": {},
                "latency_ms": row_metrics["final_row_latency_ms"],
                "usage": row_metrics["final_row_usage"],
                "system_metrics": row_metrics,
                "example_metadata": row_metadata,
                "reference": _reference_payload(row_metadata),
                "prediction": {"response_text": record.get("synthesized_text")},
                "metric_name": record.get("metric_name"),
                "metric_metadata": record.get("metadata", {}),
                "error_type": record.get("error_type"),
                "error_message": record.get("error_message"),
            })
            continue

        metric_name, primary_metric, metric_values, metric_metadata = _score_prediction(
            record.get("synthesized_text", ""),
            row_metadata,
        )
        primary_value = float(metric_values.get(primary_metric, 0.0) or 0.0)
        if source_key in per_dataset_primary:
            per_dataset_primary[source_key].append(primary_value)

        score_record = P2ScoreRecord(
            example_id=example_id,
            dataset_name=dataset_name,
            source_dataset=source_key,
            metric_name=metric_name,
            primary_metric=primary_metric,
            metrics=metric_values,
            metadata=metric_metadata,
        )
        score_records.append({
            "suite_id": run_id,
            "dataset_name": score_record.dataset_name,
            "model": P2_COUNCIL_MODEL,
            "example_id": score_record.example_id,
            "status": "scored",
            "prediction_status": "success",
            "metrics": _to_jsonable(score_record.metrics),
            "latency_ms": row_metrics["final_row_latency_ms"],
            "usage": row_metrics["final_row_usage"],
            "system_metrics": row_metrics,
            "example_metadata": row_metadata,
            "reference": _reference_payload(row_metadata),
            "prediction": {
                "response_text": record.get("synthesized_text"),
                "winning_model": record.get("winning_model"),
                "winning_role": record.get("winning_role"),
            },
            "metric_name": score_record.metric_name,
            "metric_metadata": _to_jsonable(score_record.metadata),
        })

    dataset_scores = {
        dataset: (sum(values) / len(values) if values else 0.0)
        for dataset, values in per_dataset_primary.items()
    }
    available = [dataset_scores[dataset] for dataset in _DATASET_ORDER if per_dataset_primary[dataset]]
    combined_metric = sum(available) / len(available) if available else None
    return score_records, dataset_scores, combined_metric


def _build_specialist_error_score_records(
    run_id: str,
    dataset_name: str,
    specialist_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in specialist_records:
        if record.get("status") == "success":
            continue
        row_metadata = dict(record.get("row_metadata") or {})
        row_metrics = _build_row_system_metrics(record, None)
        rows.append({
            "suite_id": run_id,
            "dataset_name": dataset_name,
            "model": P2_COUNCIL_MODEL,
            "example_id": record.get("example_id"),
            "status": "error",
            "prediction_status": "error",
            "metrics": {},
            "latency_ms": row_metrics["final_row_latency_ms"],
            "usage": row_metrics["final_row_usage"],
            "system_metrics": row_metrics,
            "example_metadata": row_metadata,
            "reference": _reference_payload(row_metadata),
            "prediction": {},
            "metric_name": _metric_name_for_row_metadata(row_metadata),
            "metric_metadata": {"phase": "specialist"},
            "error_type": record.get("error_type"),
            "error_message": record.get("error_message"),
        })
    return rows


def _build_prediction_records(
    *,
    run_id: str,
    config: P2RunConfig,
    specialist_records: list[dict[str, Any]],
    synthesized_records: list[dict[str, Any]],
    score_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    synthesized_by_id = {str(record.get("example_id")): record for record in synthesized_records}
    score_by_id = {str(record.get("example_id")): record for record in score_records}
    prediction_records: list[dict[str, Any]] = []

    for specialist in specialist_records:
        example_id = str(specialist.get("example_id"))
        row_metadata = dict(specialist.get("row_metadata") or {})
        synth = synthesized_by_id.get(example_id)
        score = score_by_id.get(example_id)
        request_payload = dict(specialist.get("request") or {})
        row_metrics = _build_row_system_metrics(specialist, synth)

        if specialist.get("status") != "success":
            prediction_records.append({
                "run_id": run_id,
                "dataset_name": config.dataset_alias,
                "model": P2_COUNCIL_MODEL,
                "example_id": example_id,
                "status": "error",
                "prompt_version": P2_PROMPT_VERSION,
                "response_text": None,
                "latency_ms": row_metrics["final_row_latency_ms"],
                "request_id": None,
                "finish_reason": None,
                "provider": config.provider,
                "usage": row_metrics["final_row_usage"],
                "example_metadata": row_metadata,
                "request_metadata": _request_metadata_payload(request_payload),
                "response_metadata": {
                    "phase_status": "specialist_failed",
                    "specialist_phase": specialist,
                    "system_metrics": row_metrics,
                },
                "error_type": specialist.get("error_type"),
                "error_message": specialist.get("error_message"),
                "raw_response": None,
            })
            continue

        if synth is None:
            prediction_records.append({
                "run_id": run_id,
                "dataset_name": config.dataset_alias,
                "model": P2_COUNCIL_MODEL,
                "example_id": example_id,
                "status": "error",
                "prompt_version": P2_PROMPT_VERSION,
                "response_text": None,
                "latency_ms": row_metrics["final_row_latency_ms"],
                "request_id": None,
                "finish_reason": None,
                "provider": config.provider,
                "usage": row_metrics["final_row_usage"],
                "example_metadata": row_metadata,
                "request_metadata": _request_metadata_payload(request_payload),
                "response_metadata": {
                    "phase_status": "missing_synth_record",
                    "specialist_phase": specialist,
                    "final_metrics": score,
                    "system_metrics": row_metrics,
                },
                "error_type": "MissingSynthRecord",
                "error_message": "No synthesizer record produced for successful specialist decision.",
                "raw_response": None,
            })
            continue

        status = "success" if synth.get("status") == "success" else "error"
        prediction_records.append({
            "run_id": run_id,
            "dataset_name": config.dataset_alias,
            "model": P2_COUNCIL_MODEL,
            "example_id": example_id,
            "status": status,
            "prompt_version": P2_PROMPT_VERSION,
            "response_text": synth.get("synthesized_text"),
            "latency_ms": row_metrics["final_row_latency_ms"],
            "request_id": synth.get("request_id"),
            "finish_reason": synth.get("finish_reason"),
            "provider": synth.get("provider") or config.provider,
            "usage": row_metrics["final_row_usage"],
            "example_metadata": row_metadata,
            "request_metadata": _request_metadata_payload(request_payload),
            "response_metadata": {
                "phase_status": "completed" if status == "success" else "synthesizer_failed",
                "specialist_phase": specialist,
                "synthesizer_phase": synth,
                "final_metrics": score,
                "system_metrics": row_metrics,
            },
            "error_type": synth.get("error_type"),
            "error_message": synth.get("error_message"),
            "raw_response": None,
        })

    return prediction_records


def _write_summaries(
    *,
    output_dir: Path,
    run_id: str,
    dataset_name: str,
    summary_state: dict[str, Any],
    dataset_scores: dict[str, float],
    combined_metric: float | None,
    phase_status: dict[str, str],
    total_examples: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    summary_files: list[Path] = []
    aggregate_rows: list[dict[str, Any]] = []

    for source_dataset in _DATASET_ORDER:
        dataset_state = summary_state["datasets"][source_dataset]
        if dataset_state["total_examples"] == 0:
            continue

        system_metrics = _system_metrics_from_state(dataset_state["system"])
        primary_metric = _primary_metric_for_source_dataset(source_dataset)
        summary_path = _summary_path(output_dir, source_dataset, P2_COUNCIL_MODEL)
        payload = {
            "suite_id": run_id,
            "dataset_name": source_dataset,
            "model": P2_COUNCIL_MODEL,
            "metric_name": primary_metric,
            "total_examples": dataset_state["total_examples"],
            "scored_examples": dataset_state["scored_examples"],
            "failed_examples": dataset_state["failed_examples"],
            "skipped_examples": 0,
            "aggregated_metrics": {primary_metric: dataset_scores.get(source_dataset, 0.0)},
            "metadata": {
                "router_dataset": dataset_name,
                "council_policy": "p2",
                "source_dataset": source_dataset,
                "phase_status": phase_status,
            },
            **system_metrics,
        }
        _write_json(summary_path, payload)
        summary_files.append(summary_path)
        aggregate_rows.append({
            "suite_id": run_id,
            "dataset_name": source_dataset,
            "model": P2_COUNCIL_MODEL,
            "primary_metric": primary_metric,
            "primary_metric_value": dataset_scores.get(source_dataset, 0.0),
            "aggregated_metrics": {primary_metric: dataset_scores.get(source_dataset, 0.0)},
            "scored_examples": dataset_state["scored_examples"],
            "failed_examples": dataset_state["failed_examples"],
            "total_examples": dataset_state["total_examples"],
            "summary_path": str(summary_path),
            "avg_latency_s": system_metrics["avg_latency_s"],
            "total_input_tokens": system_metrics["total_input_tokens"],
            "total_output_tokens": system_metrics["total_output_tokens"],
            "total_tokens": system_metrics["total_tokens"],
        })

    overall_metrics = _system_metrics_from_state(summary_state["overall"]["system"])
    overall_payload = {
        "suite_id": run_id,
        "dataset_name": dataset_name,
        "model": P2_COUNCIL_MODEL,
        "metric_name": "combined_metric",
        "total_examples": total_examples,
        "scored_examples": summary_state["overall"]["scored_examples"],
        "failed_examples": summary_state["overall"]["failed_examples"],
        "skipped_examples": 0,
        "aggregated_metrics": {"combined_metric": combined_metric, **dataset_scores},
        "metadata": {
            "source_dataset_scores": dataset_scores,
            "phase_status": phase_status,
            "council_policy": "p2",
        },
        **overall_metrics,
    }
    overall_summary_path = _summary_path(output_dir, dataset_name, P2_COUNCIL_MODEL)
    _write_json(overall_summary_path, overall_payload)
    summary_files.append(overall_summary_path)
    aggregate_rows.append({
        "suite_id": run_id,
        "dataset_name": dataset_name,
        "model": P2_COUNCIL_MODEL,
        "primary_metric": "combined_metric",
        "primary_metric_value": combined_metric,
        "aggregated_metrics": {"combined_metric": combined_metric, **dataset_scores},
        "scored_examples": summary_state["overall"]["scored_examples"],
        "failed_examples": summary_state["overall"]["failed_examples"],
        "total_examples": total_examples,
        "summary_path": str(overall_summary_path),
        "avg_latency_s": overall_metrics["avg_latency_s"],
        "total_input_tokens": overall_metrics["total_input_tokens"],
        "total_output_tokens": overall_metrics["total_output_tokens"],
        "total_tokens": overall_metrics["total_tokens"],
    })
    return summary_files, aggregate_rows


def _score_prediction(
    synthesized_text: str,
    row_metadata: dict[str, Any],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    source_key = _canonical_source_dataset(row_metadata.get("source_dataset", ""))
    gold_answers = _gold_answers(row_metadata, source_key)

    # --- THE FIX: Strip out all reasoning/chain-of-thought blocks ---
    clean_text = synthesized_text
    
    # Handle DeepSeek/Qwen Reasoning tags
    if "</think>" in clean_text:
        clean_text = clean_text.split("</think>")[-1]
    
    # Handle the custom MuSiQue scratchpad tags from your prompts.py
    if "</scratchpad>" in clean_text:
        clean_text = clean_text.split("</scratchpad>")[-1]
        
    clean_text = clean_text.strip()
    # -----------------------------------------------------------------

    if source_key == "musique":
        answerable = bool((row_metadata.get("parsed_metadata") or {}).get("answerable", False))
        references = gold_answers if answerable else [_MUSIQUE_NOT_PRESENT]
        # Pass the clean_text, NOT the raw synthesized_text
        extracted = extract_qa_answer_musique(clean_text)
        return (
            "musique_token_f1",
            "token_f1",
            {
                "exact_match": exact_match_multi(extracted, references),
                "token_f1": token_f1_multi(extracted, references),
            },
            {"extracted_answer": extracted, "answerable": answerable},
        )

    if source_key == "quality":
        extracted = extract_qa_answer(clean_text)
        return (
            "token_f1",
            "token_f1",
            {
                "exact_match": exact_match_multi(extracted, gold_answers),
                "token_f1": token_f1_multi(extracted, gold_answers),
            },
            {"extracted_answer": extracted},
        )

    if source_key == "fever":
        extracted = extract_fever_label(clean_text)
        reference = str(row_metadata.get("gold_label") or "")
        return (
            "label_accuracy",
            "label_accuracy",
            {"label_accuracy": label_accuracy(extracted, reference)},
            {"extracted_answer": extracted},
        )

    if source_key == "hardmath":
        extracted = extract_math_answer(clean_text)
        reference = str(row_metadata.get("gold_answer") or "")
        return (
            "math_exact_match",
            "math_exact_match",
            {"math_exact_match": math_exact_match(extracted, reference)},
            {"extracted_answer": extracted},
        )

    # HumanEval Plus fallback
    extracted = extract_code_answer(clean_text)
    return (
        "pass_at_1",
        "pass_at_1",
        {
            "pass_at_1": pass_at_1(
                extracted,
                test_code=str(row_metadata.get("unit_tests") or ""),
                entry_point=str(row_metadata.get("entry_point") or ""),
            )
        },
        {"extracted_answer": extracted},
    )

from collections import defaultdict

# def _iter_dataset_rows(config: P2RunConfig) -> Iterator[dict[str, Any]]:
#     dataset = load_dataset(config.dataset_name, split=config.split, streaming=False)

#     # Fast, vectorized filter
#     dataset = dataset.filter(lambda x: x["source_dataset"] == "QuALITY")


#     if config.max_examples is not None:
#         dataset = dataset.select(range(min(len(dataset), config.max_examples)))

#     # iterate normally (already filtered)
#     yield from dataset
    

def _iter_dataset_rows(config):
    dataset = load_dataset(config.dataset_name, split=config.split)

    seen = defaultdict(list)

    # First pass: group rows (lightweight if dataset fits in memory)
    for row in dataset:
        seen[row["source_dataset"]].append(row)

    keys = list(seen.keys())

    count = 0
    max_n = config.max_examples or float("inf")
    i = 0

    # Round-robin: 1 of each
    while count < max_n:
        for k in keys:
            if i < len(seen[k]):
                yield seen[k][i]
                count += 1

                if count >= max_n:
                    return
        i += 1

def _batched_rows(iterator: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    print(f"_batched_rows {batch_size}")
    batch: list[dict[str, Any]] = []
    for row in iterator:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    print(f"_batched_rows done {batch_size}")
    if batch:
        yield batch


def _build_specialist_orchestrator_config(config: P2RunConfig):
    if config.provider == "vllm":
        return build_default_local_vllm_orchestrator_config()
    return build_default_orchestrator_config(
        provider=config.provider,
        api_base=config.api_base or P2_API_BASE,
        api_key_env=config.api_key_env or P2_API_KEY_ENV,
        qa_model=P2_API_QA_MODEL,
        reasoning_model=P2_API_REASONING_MODEL,
        general_model=P2_API_GENERAL_MODEL,
    )


def _build_synthesizer_orchestrator_config(config: P2RunConfig) -> OrchestratorConfig:
    synth_model = config.synth_model or P2_API_SYNTH_MODEL
    if config.provider == "vllm":
        synth_model = P2_VLLM_SYNTH_MODEL
        return OrchestratorConfig(
            models=(
                ModelSpec(
                    role="synthesizer",
                    model=synth_model,
                    aliases=("synthesizer",),
                    provider_config=ProviderConfig(
                        provider=Provider.LOCAL,
                        default_model=synth_model,
                        default_params={
                            "local_launch_image": LocalVLLMPresetConfig().image,
                            "local_launch_port": config.local_launch_port + 10,
                            "local_launch_bind": DEFAULT_LOCAL_VLLM_BIND,
                            "local_launch_startup_timeout_seconds": 1800.0,
                            "local_launch_quantization": "compressed-tensors",
                            "local_launch_use_gpu": True,
                        },
                    ),
                ),
            ),
            default_role="synthesizer",
            mode_label="local",
        )

    return OrchestratorConfig(
        models=(
            ModelSpec(
                role="synthesizer",
                model=synth_model,
                aliases=("synthesizer",),
                provider_config=ProviderConfig(
                    provider=config.provider,
                    api_base=config.api_base or P2_API_BASE,
                    api_key_env=config.api_key_env or P2_API_KEY_ENV,
                    default_model=synth_model,
                    timeout_seconds=60,
                    max_retries=1,
                ),
            ),
        ),
        default_role="synthesizer",
        mode_label="http",
    )


def _build_request_for_row(
    config: P2RunConfig,
    index: int,
    row: dict[str, Any],
) -> tuple[str, PromptRequest, dict[str, Any]]:
    example_id = str(row.get("id", index))
    source_dataset = _canonical_source_dataset(str(row.get("source_dataset", "")))
    skill_tags = list(row.get("skill_tags") or [])
    question = str(row.get("question", ""))
    context = row.get("context") or None

    if source_dataset == "musique":
        context = _extract_musique_oracle_context(row, context)

    parsed_metadata = _parse_metadata_field(row.get("metadata"))
    specialist_prompt, specialist_context = build_specialist_prompt(
        source_dataset=source_dataset,
        question=question,
        context=context,
        skill_tags=skill_tags,
    )
    row_metadata = {
        "source_dataset": source_dataset,
        "skill_tags": skill_tags,
        "gold_answer": row.get("gold_answer"),
        "gold_label": row.get("gold_label"),
        "unit_tests": row.get("unit_tests"),
        "entry_point": row.get("entry_point") or parsed_metadata.get("entry_point"),
        "parsed_metadata": parsed_metadata,
        "original_id": row.get("original_id"),
        "question": question,
        "context": context,
        "specialist_prompt": specialist_prompt,
    }
    request = PromptRequest(
        user_prompt=specialist_prompt,
        context=specialist_context,
        metadata={
            "example_id": example_id,
            "dataset_name": config.dataset_alias,
            **row_metadata,
        },
    )
    return example_id, request, row_metadata


def _parse_metadata_field(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_musique_oracle_context(row: dict[str, Any], fallback: str | None) -> str | None:
    raw_context = row.get("paragraphs", row.get("context"))
    best_paragraphs: list[str] = []

    if isinstance(raw_context, list):
        for item in raw_context:
            if isinstance(item, dict) and item.get("is_supporting") is True:
                text_chunk = item.get("paragraph_text", item.get("text", ""))
                if text_chunk.strip():
                    best_paragraphs.append(text_chunk.strip())
        if not best_paragraphs:
            for item in raw_context[:4]:
                if isinstance(item, dict):
                    text_chunk = item.get("paragraph_text", item.get("text", ""))
                    if text_chunk.strip():
                        best_paragraphs.append(text_chunk.strip())
    elif isinstance(raw_context, str):
        best_paragraphs = [part.strip() for part in raw_context.split("\n\n") if part.strip()][:4]

    if best_paragraphs:
        return "\n\n".join(best_paragraphs)
    return fallback


def _serialize_decision(
    decision: P2CouncilDecision,
    *,
    row_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "example_id": decision.example_id,
        "dataset_name": decision.dataset_name,
        "status": "success",
        "request": _to_jsonable(decision.request),
        "row_metadata": row_metadata,
        "winning_label": decision.winning_label,
        "winning_role": decision.winning_role,
        "winning_model": decision.winning_model,
        "winning_answer": decision.winning_answer,
        "votes": _to_jsonable(decision.votes),
        "aggregated_usage": _to_jsonable(decision.aggregated_usage),
        "system_metrics": _to_jsonable(decision.system_metrics),
        "metadata": _to_jsonable(decision.metadata),
        "role_results": {
            role: _to_jsonable(role_result)
            for role, role_result in decision.role_results.items()
        },
    }


def _response_usage(response: Any) -> Usage:
    prompt_response = getattr(response, "prompt_response", None)
    usage = getattr(prompt_response, "usage", None)
    return usage if isinstance(usage, Usage) else Usage()


def _reference_payload(row_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "gold_answer": row_metadata.get("gold_answer"),
        "gold_label": row_metadata.get("gold_label"),
        "unit_tests": row_metadata.get("unit_tests"),
        "entry_point": row_metadata.get("entry_point"),
        "parsed_metadata": row_metadata.get("parsed_metadata"),
    }


def _request_metadata_payload(request_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "system_prompt": request_payload.get("system_prompt"),
        "temperature": request_payload.get("temperature"),
        "context": request_payload.get("context"),
        "metadata": request_payload.get("metadata"),
    }


def _combine_usage_dicts(first: dict[str, Any] | None, second: dict[str, Any] | None) -> dict[str, Any]:
    first = first or {}
    second = second or {}

    def _add(name: str) -> int | None:
        left = first.get(name)
        right = second.get(name)
        if left is None and right is None:
            return None
        return int(left or 0) + int(right or 0)

    return {
        "input_tokens": _add("input_tokens"),
        "output_tokens": _add("output_tokens"),
        "total_tokens": _add("total_tokens"),
        "cost": None,
        "currency": first.get("currency") or second.get("currency"),
    }


def _role_result_usage(role_results: dict[str, Any], role: str) -> dict[str, Any]:
    role_payload = dict(role_results.get(role) or {})
    return dict(role_payload.get("usage") or {})


def _role_result_latency(role_results: dict[str, Any], role: str) -> float | None:
    role_payload = dict(role_results.get(role) or {})
    return _as_float_or_none(role_payload.get("latency_ms"))


def _build_row_system_metrics(
    specialist: dict[str, Any] | None,
    synth: dict[str, Any] | None,
) -> dict[str, Any]:
    specialist = specialist or {}
    synth = synth or {}
    specialist_metrics = dict(specialist.get("system_metrics") or {})
    role_results = dict(specialist.get("role_results") or {})
    specialist_usage = dict(specialist_metrics.get("specialist_usage") or specialist.get("aggregated_usage") or {})
    synth_usage = dict(synth.get("usage") or {})
    specialist_latency_ms = _as_float_or_none(
        specialist_metrics.get("specialist_latency_ms_total", specialist_metrics.get("wall_clock_latency_ms"))
    )
    synth_latency_ms = _as_float_or_none(synth.get("latency_ms"))

    qa_usage = _role_result_usage(role_results, "qa")
    reasoning_usage = _role_result_usage(role_results, "reasoning")
    general_usage = _role_result_usage(role_results, "general")
    qa_latency_ms = _role_result_latency(role_results, "qa")
    reasoning_latency_ms = _role_result_latency(role_results, "reasoning")
    general_latency_ms = _role_result_latency(role_results, "general")

    return {
        "qa_usage": qa_usage,
        "reasoning_usage": reasoning_usage,
        "general_usage": general_usage,
        "qa_latency_ms": qa_latency_ms,
        "reasoning_latency_ms": reasoning_latency_ms,
        "general_latency_ms": general_latency_ms,
        "answer_usage_total": dict(specialist_metrics.get("answer_usage") or {}),
        "vote_usage_total": dict(specialist_metrics.get("vote_usage") or {}),
        "specialist_usage_total": specialist_usage,
        "synthesizer_usage": synth_usage,
        "final_row_usage": specialist_usage,
        "answer_latency_ms_total": _as_float_or_none(specialist_metrics.get("answer_latency_ms_total")),
        "vote_latency_ms_total": _as_float_or_none(specialist_metrics.get("vote_latency_ms_total")),
        "specialist_latency_ms_total": specialist_latency_ms,
        "wall_clock_latency_ms": _as_float_or_none(specialist_metrics.get("wall_clock_latency_ms")),
        "synthesizer_latency_ms": synth_latency_ms,
        "final_row_latency_ms": specialist_latency_ms,
    }


def _metric_name_for_row_metadata(row_metadata: dict[str, Any]) -> str:
    source_key = _canonical_source_dataset(row_metadata.get("source_dataset", ""))
    return {
        "musique": "musique_token_f1",
        "quality": "token_f1",
        "fever": "label_accuracy",
        "hardmath": "math_exact_match",
        "humaneval_plus": "pass_at_1",
    }[source_key]


def _primary_metric_for_source_dataset(source_dataset: str) -> str:
    source_key = _canonical_source_dataset(source_dataset)
    return {
        "musique": "token_f1",
        "quality": "token_f1",
        "fever": "label_accuracy",
        "hardmath": "math_exact_match",
        "humaneval_plus": "pass_at_1",
    }[source_key]


def _canonical_source_dataset(source_dataset: str) -> str:
    normalized = (source_dataset or "").strip().lower()
    if "musique" in normalized:
        return "musique"
    if "quality" in normalized or "narrativeqa" in normalized:
        return "quality"
    if "fever" in normalized:
        return "fever"
    if "hardmath" in normalized or "math" in normalized:
        return "hardmath"
    return "humaneval_plus"


def _gold_answers(row_metadata: dict[str, Any], source_key: str) -> list[str]:
    raw = row_metadata.get("gold_answer")
    if raw is None:
        return []
    if source_key == "quality":
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (ValueError, SyntaxError):
            return [str(raw)]

        answers = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "text" in item:
                    answers.append(str(item["text"]))
                else:
                    answers.append(str(item))
            if answers:
                return answers
        return [str(raw)]

    if isinstance(raw, list):
        answers = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                answers.append(str(item["text"]))
            else:
                answers.append(str(item))
        return answers
    return [str(raw)]


def _new_summary_state() -> dict[str, Any]:
    return {
        "datasets": {name: _new_dataset_summary_state() for name in _DATASET_ORDER},
        "overall": _new_dataset_summary_state(),
    }


def _new_dataset_summary_state() -> dict[str, Any]:
    return {
        "total_examples": 0,
        "scored_examples": 0,
        "failed_examples": 0,
        "primary_metric_sum": 0.0,
        "primary_metric_count": 0,
        "system": {
            "successful_rows": 0,
            "total_latency_ms": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
        },
    }


def _accumulate_summary_state(
    summary_state: dict[str, Any],
    prediction_records: list[dict[str, Any]],
    score_records: list[dict[str, Any]],
) -> None:
    for record in score_records:
        row_metadata = dict(record.get("example_metadata") or {})
        source_dataset = _canonical_source_dataset(str(row_metadata.get("source_dataset", "")))
        dataset_state = summary_state["datasets"][source_dataset]
        dataset_state["total_examples"] += 1
        summary_state["overall"]["total_examples"] += 1
        if record.get("status") == "scored":
            dataset_state["scored_examples"] += 1
            summary_state["overall"]["scored_examples"] += 1
            primary_metric = _primary_metric_for_source_dataset(source_dataset)
            dataset_state["primary_metric_sum"] += float((record.get("metrics") or {}).get(primary_metric, 0.0) or 0.0)
            dataset_state["primary_metric_count"] += 1
        else:
            dataset_state["failed_examples"] += 1
            summary_state["overall"]["failed_examples"] += 1

    for record in prediction_records:
        if record.get("status") != "success":
            continue
        row_metadata = dict(record.get("example_metadata") or {})
        source_dataset = _canonical_source_dataset(str(row_metadata.get("source_dataset", "")))
        _accumulate_system(summary_state["datasets"][source_dataset]["system"], record)
        _accumulate_system(summary_state["overall"]["system"], record)


def _accumulate_system(state: dict[str, Any], prediction_record: dict[str, Any]) -> None:
    usage = dict(prediction_record.get("usage") or {})
    state["successful_rows"] += 1
    state["total_latency_ms"] += float(prediction_record.get("latency_ms") or 0.0)
    state["total_input_tokens"] += int(usage.get("input_tokens") or 0)
    state["total_output_tokens"] += int(usage.get("output_tokens") or 0)
    state["total_tokens"] += int(usage.get("total_tokens") or 0)


def _compute_dataset_scores(summary_state: dict[str, Any]) -> tuple[dict[str, float], float | None]:
    dataset_scores = {}
    for dataset in _DATASET_ORDER:
        state = summary_state["datasets"][dataset]
        count = state["primary_metric_count"]
        dataset_scores[dataset] = (state["primary_metric_sum"] / count) if count else 0.0
    available = [dataset_scores[name] for name in _DATASET_ORDER if summary_state["datasets"][name]["primary_metric_count"]]
    return dataset_scores, (sum(available) / len(available) if available else None)


def _system_metrics_from_state(state: dict[str, Any]) -> dict[str, Any]:
    count = state["successful_rows"]
    total_latency_ms = state["total_latency_ms"]
    total_input_tokens = state["total_input_tokens"]
    total_output_tokens = state["total_output_tokens"]
    total_tokens = state["total_tokens"]
    return {
        "avg_latency_s": (total_latency_ms / count / 1000.0) if count else None,
        "avg_latency_ms": (total_latency_ms / count) if count else None,
        "total_input_tokens": total_input_tokens if count else 0,
        "total_output_tokens": total_output_tokens if count else 0,
        "total_tokens": total_tokens if count else 0,
        "avg_input_tokens": (total_input_tokens / count) if count else None,
        "avg_output_tokens": (total_output_tokens / count) if count else None,
        "avg_total_tokens": (total_tokens / count) if count else None,
    }


def _prediction_path(output_dir: Path, dataset_name: str, model: str) -> Path:
    return output_dir / "predictions" / f"{_pair_run_name(dataset_name, model)}.jsonl"


def _score_path(output_dir: Path, dataset_name: str, model: str) -> Path:
    return output_dir / "scores" / f"{_pair_run_name(dataset_name, model)}.jsonl"


def _summary_path(output_dir: Path, dataset_name: str, model: str) -> Path:
    return output_dir / "summaries" / f"{_pair_run_name(dataset_name, model)}.json"


def _pair_run_name(dataset_name: str, model: str) -> str:
    return f"{_slugify(dataset_name)}__{_slugify(model)}"


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("_") or "value"


def _reset_jsonl_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(row), sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_manifest(
    path: Path,
    *,
    config: P2RunConfig,
    run_id: str,
    total_examples: int,
    phase_status: dict[str, str],
    processed_examples: int,
    completed_examples: int,
    failed_examples: int,
    current_batch_index: int,
    partial_dataset_scores: dict[str, float] | None = None,
    partial_combined_metric: float | None = None,
    result: P2RunResult | None = None,
) -> None:
    payload = {
        "run_id": run_id,
        "config": _to_jsonable(config),
        "dataset": {
            "name": config.dataset_name,
            "alias": config.dataset_alias,
            "split": config.split,
            "total_examples": total_examples,
            "processed_examples": processed_examples,
            "batch_size": config.batch_size,
            "current_batch_index": current_batch_index,
        },
        "progress": {
            "completed_examples": completed_examples,
            "failed_examples": failed_examples,
        },
        "phase_status": phase_status,
    }
    if partial_dataset_scores is not None:
        payload["partial_dataset_scores"] = partial_dataset_scores
    if partial_combined_metric is not None:
        payload["partial_combined_metric"] = partial_combined_metric
    if result is not None:
        payload["result"] = _to_jsonable(result)
    _write_json(path, payload)


def _sum_float_values(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) if numeric else None


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("p2_run_%Y%m%dT%H%M%SZ")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Usage):
        return {
            "input_tokens": value.input_tokens,
            "output_tokens": value.output_tokens,
            "total_tokens": value.total_tokens,
            "cost": value.cost,
            "currency": value.currency,
        }
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
