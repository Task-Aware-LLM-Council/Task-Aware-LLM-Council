from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from llm_gateway import BaseLLMClient, PromptResponse, create_client

from benchmarking_pipeline.models import (
    BenchmarkDataset,
    BenchmarkExample,
    BenchmarkPrediction,
    BenchmarkRunConfig,
    BenchmarkRunResult,
    default_run_id,
)
from benchmarking_pipeline.prompts import build_prompt_request
from benchmarking_pipeline.storage import (
    append_prediction,
    ensure_run_directory,
    load_recorded_example_ids,
    prediction_path,
    write_manifest,
)


async def run_benchmark(
    datasets: Iterable[BenchmarkDataset],
    config: BenchmarkRunConfig,
    *,
    client: BaseLLMClient | None = None,
    client_factory: Callable = create_client,
) -> BenchmarkRunResult:
    dataset_list = tuple(datasets)
    run_id = config.run_name or default_run_id()
    run_dir = ensure_run_directory(Path(config.output_root), run_id)
    manifest_path = write_manifest(
        run_dir,
        run_id=run_id,
        config=config,
        datasets=dataset_list,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    prediction_files: set[Path] = set()
    total_examples = sum(len(dataset.examples) for dataset in dataset_list) * len(config.models)
    attempted_examples = 0
    completed_examples = 0
    failed_examples = 0
    skipped_existing = 0

    created_client = client is None
    active_client = client or client_factory(config.provider_config)

    try:
        for dataset in dataset_list:
            for model in config.models:
                path = prediction_path(run_dir, dataset.name, model)
                prediction_files.add(path)

                completed_ids = (
                    load_recorded_example_ids(path) if config.skip_existing else set()
                )
                examples_to_run = [
                    example
                    for example in dataset.examples
                    if example.example_id not in completed_ids
                ]
                skipped_existing += len(dataset.examples) - len(examples_to_run)
                if not examples_to_run:
                    continue

                semaphore = asyncio.Semaphore(max(config.max_concurrency, 1))
                write_lock = asyncio.Lock()

                async def _run_example(example: BenchmarkExample) -> BenchmarkPrediction:
                    async with semaphore:
                        prediction = await _generate_prediction(
                            active_client,
                            config=config,
                            example=example,
                            model=model,
                            run_id=run_id,
                        )

                    async with write_lock:
                        append_prediction(path, prediction)

                    return prediction

                tasks = [asyncio.create_task(_run_example(example)) for example in examples_to_run]
                try:
                    batch_predictions = await asyncio.gather(*tasks)
                except Exception:
                    for task in tasks:
                        task.cancel()
                    raise

                attempted_examples += len(batch_predictions)
                completed_examples += sum(
                    1 for prediction in batch_predictions if prediction.status == "success"
                )
                failed_examples += sum(
                    1 for prediction in batch_predictions if prediction.status == "error"
                )
    finally:
        if created_client:
            await active_client.close()

    result = BenchmarkRunResult(
        run_id=run_id,
        output_dir=run_dir,
        manifest_path=manifest_path,
        created_at=created_at,
        total_examples=total_examples,
        attempted_examples=attempted_examples,
        completed_examples=completed_examples,
        failed_examples=failed_examples,
        skipped_existing=skipped_existing,
        prediction_files=tuple(sorted(prediction_files)),
    )
    manifest_path = write_manifest(
        run_dir,
        run_id=run_id,
        config=config,
        datasets=dataset_list,
        result=result,
    )
    return BenchmarkRunResult(
        run_id=result.run_id,
        output_dir=result.output_dir,
        manifest_path=manifest_path,
        created_at=result.created_at,
        total_examples=result.total_examples,
        attempted_examples=result.attempted_examples,
        completed_examples=result.completed_examples,
        failed_examples=result.failed_examples,
        skipped_existing=result.skipped_existing,
        prediction_files=result.prediction_files,
    )


async def _generate_prediction(
    client: BaseLLMClient,
    *,
    config: BenchmarkRunConfig,
    example: BenchmarkExample,
    model: str,
    run_id: str,
) -> BenchmarkPrediction:
    request = build_prompt_request(example, model=model, config=config)

    try:
        response = await client.generate(request)
    except Exception as exc:
        if not config.continue_on_error:
            raise

        return BenchmarkPrediction(
            run_id=run_id,
            dataset_name=example.dataset_name,
            model=model,
            example_id=example.example_id,
            status="error",
            prompt_version=config.prompt_version,
            example_metadata=dict(example.metadata),
            request_metadata=dict(request.metadata),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    return _prediction_from_response(
        run_id=run_id,
        prompt_version=config.prompt_version,
        example=example,
        model=model,
        request_metadata=dict(request.metadata),
        response=response,
        save_raw_response=config.save_raw_response,
    )


def _prediction_from_response(
    *,
    run_id: str,
    prompt_version: str,
    example: BenchmarkExample,
    model: str,
    request_metadata: dict[str, object],
    response: PromptResponse,
    save_raw_response: bool,
) -> BenchmarkPrediction:
    usage = asdict(response.usage)
    raw_response = response.raw_response if save_raw_response else None

    return BenchmarkPrediction(
        run_id=run_id,
        dataset_name=example.dataset_name,
        model=model,
        example_id=example.example_id,
        status="success",
        prompt_version=prompt_version,
        response_text=response.text,
        latency_ms=response.latency_ms,
        request_id=response.request_id,
        finish_reason=response.finish_reason,
        provider=response.provider,
        usage=usage,
        example_metadata=dict(example.metadata),
        request_metadata=request_metadata,
        response_metadata=dict(response.metadata),
        raw_response=raw_response,
    )
