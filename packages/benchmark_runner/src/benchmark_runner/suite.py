from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable
import asyncio
import time

from benchmarking_pipeline import (
    BenchmarkDataset,
    BenchmarkRunConfig,
    BenchmarkRunResult,
    run_benchmark,
)
from task_eval import DatasetProfile, get_dataset_profile
from llm_gateway import managed_local_provider_config

from benchmark_runner.metrics import Metric, NullMetric, aggregate_score_records
from benchmark_runner.models import (
    AggregateMetricRow,
    BenchmarkCase,
    BenchmarkSpec,
    BenchmarkSuiteResult,
    DatasetRunConfig,
    ScoreRecord,
    ScoreSummary,
    default_suite_id,
)
from benchmark_runner.sources import DatasetSource, chunk_cases
from benchmark_runner.storage import (
    aggregate_summary_path,
    append_score_record,
    ensure_suite_directory,
    pair_run_name,
    read_prediction_records,
    score_path,
    summary_path,
    write_aggregate_summary,
    write_manifest,
    write_summary,
)


MetricResolver = Callable[[DatasetSource], Metric]
PipelineRunner = Callable[[
    Iterable[BenchmarkDataset], BenchmarkRunConfig], object]


async def run_benchmark_suite(
    dataset_sources: Iterable[DatasetSource],
    spec: BenchmarkSpec,
    *,
    metric_resolver: MetricResolver | None = None,
    pipeline_runner=run_benchmark,
    delay_between_runs=5,
    client_stats_provider=None,  # callable() -> dict, called after all batches for a source
) -> BenchmarkSuiteResult:
    sources = tuple(dataset_sources)
    suite_id = spec.suite_name or default_suite_id()
    suite_dir = ensure_suite_directory(Path(spec.output_root), suite_id)
    datasets_manifest = [
        {"name": source.name, "metadata": dict(
            getattr(source, "metadata", {}))}
        for source in sources
    ]
    manifest_path = write_manifest(
        suite_dir,
        suite_id=suite_id,
        spec=spec,
        datasets=datasets_manifest,
    )

    metric_resolver = metric_resolver or (lambda source: NullMetric())

    score_files: set[Path] = set()
    summary_files: set[Path] = set()
    aggregate_rows: list[AggregateMetricRow] = []
    total_pairs = 0
    total_examples = 0
    scored_examples = 0
    failed_examples = 0

    for model in spec.models:
        async with managed_local_provider_config(spec.provider_config, model=model) as provider_config:
            for source in sources:
                batch_size = spec.batch_size
                max_concurrency = spec.max_concurrency
                if source.name == "hardmath" or source.name == "humaneval_plus":
                #     batch_size = 15
                    max_concurrency = 60
                print(f"Started running for model:{model} - source:{source.name} - batch_size:{batch_size} - max_concurrency:{max_concurrency}")
                start_time = time.perf_counter()  # START TIMER
                dataset_metadata = dict(getattr(source, "metadata", {}))
                metric = metric_resolver(source)
                pair_score_path = score_path(suite_dir, source.name, model)
                pair_summary_path = summary_path(suite_dir, source.name, model)
                score_files.add(pair_score_path)
                summary_files.add(pair_summary_path)

                pair_records: list[dict[str, object]] = []

                iterations = 0
                for batch in (
                    chunk_cases(
                        source,
                        batch_size=max(batch_size, 1),
                        max_cases=spec.max_examples_per_dataset,
                    )
                ):  
                    print(f"running batch-{iterations}")
                    await asyncio.sleep(spec.delay_between_requests)
                    batch_examples = tuple(case.example for case in batch)
                    case_lookup = {case.example.example_id: case for case in batch}
                    total_examples += len(batch_examples)

                    pair_name = pair_run_name(source.name, model)
                    pipeline_config = BenchmarkRunConfig(
                        provider_config=provider_config,
                        models=(model,),
                        output_root=suite_dir / "predictions",
                        run_name=pair_name,
                        prompt_version=spec.prompt_version,
                        max_concurrency=max_concurrency,
                        continue_on_error=spec.continue_on_error,
                        skip_existing=spec.skip_existing_predictions,
                        save_raw_response=spec.save_raw_response,
                        temperature=spec.temperature,
                        max_tokens=spec.max_tokens,
                        stop_sequences=spec.stop_sequences,
                        provider_params=dict(spec.provider_params),
                    )

                    print(f"Running pipeline - {iterations}")
                    result = await pipeline_runner(
                        (BenchmarkDataset(name=source.name, examples=batch_examples),),
                        pipeline_config,
                    )
                    print(f"Pipeline done - {iterations}")

                    print(f"loading predictions - {iterations}")
                    batch_prediction_records = _load_batch_predictions(
                        result,
                        dataset_name=source.name,
                        model=model,
                        example_ids=set(case_lookup),
                    )
                    print(f"predictions loaded - {iterations}")

                    print(f"scoring predictions - {iterations}")
                    for prediction in batch_prediction_records:
                        record = _score_prediction(
                            suite_id=suite_id,
                            dataset_name=source.name,
                            model=model,
                            metric=metric,
                            case_lookup=case_lookup,
                            dataset_metadata=dataset_metadata,
                            prediction=prediction,
                        )
                        append_score_record(pair_score_path, record)
                        pair_records.append(asdict(record))
                        if record.status == "scored":
                            scored_examples += 1
                        else:
                            failed_examples += 1
                    
                    print(f"predictions scored - {iterations}")

                    print(f"batch done -{iterations}")

                    iterations += 1
                    if iterations%10==0:
                        print(f"model:{model} - source:{source.name} iterations:{iterations} done")
                    
                end_time = time.perf_counter()    # END TIMER
                elapsed = end_time - start_time
                print(f"Finished running for model:{model} - source:{source.name} in {elapsed:.2f} seconds")
                total_pairs += 1

                client_stats = client_stats_provider() if client_stats_provider else None
                summary = _build_summary(
                    suite_id=suite_id,
                    dataset_name=source.name,
                    model=model,
                    metric_name=_metric_name(metric, source),
                    records=pair_records,
                    client_stats=client_stats,
                )
                write_summary(pair_summary_path, summary)
                aggregate_rows.append(
                    AggregateMetricRow(
                        suite_id=suite_id,
                        dataset_name=summary.dataset_name,
                        model=summary.model,
                        primary_metric=summary.metric_name,
                        primary_metric_value=summary.aggregated_metrics.get(
                            summary.metric_name),
                        aggregated_metrics=dict(summary.aggregated_metrics),
                        scored_examples=summary.scored_examples,
                        failed_examples=summary.failed_examples,
                        total_examples=summary.total_examples,
                        summary_path=pair_summary_path,
                        avg_latency_s=summary.avg_latency_s,
                        total_input_tokens=summary.total_input_tokens,
                        total_output_tokens=summary.total_output_tokens,
                        total_tokens=summary.total_tokens,
                    )
                )

                _print_dataset_summary(source.name, model, summary)
                print(f"Done running for model:{model} - source:{source.name}")

    aggregate_path = aggregate_summary_path(suite_dir)
    write_aggregate_summary(aggregate_path, aggregate_rows)

    suite_result = BenchmarkSuiteResult(
        suite_id=suite_id,
        output_dir=suite_dir,
        manifest_path=manifest_path,
        score_files=tuple(sorted(score_files)),
        summary_files=tuple(sorted(summary_files)),
        aggregate_summary_path=aggregate_path,
        total_pairs=total_pairs,
        total_examples=total_examples,
        scored_examples=scored_examples,
        failed_examples=failed_examples,
    )
    manifest_path = write_manifest(
        suite_dir,
        suite_id=suite_id,
        spec=spec,
        datasets=datasets_manifest,
        result=suite_result,
    )
    return BenchmarkSuiteResult(
        suite_id=suite_result.suite_id,
        output_dir=suite_result.output_dir,
        manifest_path=manifest_path,
        score_files=suite_result.score_files,
        summary_files=suite_result.summary_files,
        aggregate_summary_path=suite_result.aggregate_summary_path,
        total_pairs=suite_result.total_pairs,
        total_examples=suite_result.total_examples,
        scored_examples=suite_result.scored_examples,
        failed_examples=suite_result.failed_examples,
    )


def _load_batch_predictions(
    result: BenchmarkRunResult,
    *,
    dataset_name: str,
    model: str,
    example_ids: set[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for path in result.prediction_files:
        if path.name != f"{pair_run_name(dataset_name, model)}.jsonl":
            continue

        for record in read_prediction_records(path):
            example_id = record.get("example_id")
            if isinstance(example_id, str) and example_id in example_ids:
                records.append(record)

    return records


def _score_prediction(
    *,
    suite_id: str,
    dataset_name: str,
    model: str,
    metric: Metric,
    case_lookup: dict[str, BenchmarkCase],
    dataset_metadata: dict[str, object],
    prediction: dict[str, object],
) -> ScoreRecord:
    example_id = str(prediction.get("example_id"))
    prediction_status = str(prediction.get("status", "unknown"))
    case = case_lookup[example_id]

    if prediction_status != "success":
        return ScoreRecord(
            suite_id=suite_id,
            dataset_name=dataset_name,
            model=model,
            example_id=example_id,
            status="prediction_error",
            prediction_status=prediction_status,
            example_metadata=dict(case.example.metadata),
            reference=dict(case.reference),
            prediction=prediction,
            metric_name=metric.name,
            error_type=str(prediction.get("error_type") or "PredictionError"),
            error_message=str(prediction.get("error_message")
                              or "prediction failed"),
        )

    try:
        result = metric.score(
            case=case,
            prediction=prediction,
            dataset_metadata=dataset_metadata,
        )
    except Exception as exc:
        return ScoreRecord(
            suite_id=suite_id,
            dataset_name=dataset_name,
            model=model,
            example_id=example_id,
            status="metric_error",
            prediction_status=prediction_status,
            example_metadata=dict(case.example.metadata),
            reference=dict(case.reference),
            prediction=prediction,
            metric_name=metric.name,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    return ScoreRecord(
        suite_id=suite_id,
        dataset_name=dataset_name,
        model=model,
        example_id=example_id,
        status="scored",
        prediction_status=prediction_status,
        metrics=dict(result.values),
        example_metadata=dict(case.example.metadata),
        reference=dict(case.reference),
        prediction=prediction,
        metric_name=metric.name,
        metric_metadata=dict(result.metadata),
    )


def _build_summary(
    *,
    suite_id: str,
    dataset_name: str,
    model: str,
    metric_name: str,
    records: list[dict[str, object]],
    client_stats: dict | None = None,
) -> ScoreSummary:
    scored_records = [
        record for record in records if record.get("status") == "scored"]
    failed_records = [
        record for record in records if record.get("status") != "scored"]

    stats = client_stats or {}
    return ScoreSummary(
        suite_id=suite_id,
        dataset_name=dataset_name,
        model=model,
        metric_name=metric_name,
        total_examples=len(records),
        scored_examples=len(scored_records),
        failed_examples=len(failed_records),
        skipped_examples=0,
        aggregated_metrics=aggregate_score_records(scored_records),
        avg_latency_s=stats.get("avg_latency_s"),
        total_input_tokens=stats.get("total_input_tokens"),
        total_output_tokens=stats.get("total_output_tokens"),
        total_tokens=stats.get("total_tokens"),
    )


def _print_dataset_summary(dataset_name: str, model: str, summary: "ScoreSummary") -> None:
    acc = summary.aggregated_metrics.get(summary.metric_name)
    acc_str = f"{acc:.4f}" if acc is not None else "n/a"
    lat_str = f"{summary.avg_latency_s:.1f}s" if summary.avg_latency_s is not None else "n/a"
    tok_str = str(summary.total_tokens) if summary.total_tokens is not None else "n/a"
    in_str = str(summary.total_input_tokens) if summary.total_input_tokens is not None else "n/a"
    out_str = str(summary.total_output_tokens) if summary.total_output_tokens is not None else "n/a"
    print(
        f"\n=== Dataset Results: {dataset_name} | model: {model} ===\n"
        f"  accuracy ({summary.metric_name}): {acc_str}\n"
        f"  scored/total examples:            {summary.scored_examples}/{summary.total_examples}\n"
        f"  avg latency per example:          {lat_str}\n"
        f"  total tokens:                     {tok_str}  (input={in_str}, output={out_str})\n"
    )


def _metric_name(metric: Metric, source: DatasetSource) -> str:
    primary_metric = getattr(metric, "primary_metric", None)
    if isinstance(primary_metric, str) and primary_metric:
        return primary_metric
    name = getattr(metric, "name", None)
    if isinstance(name, str) and name:
        return name
    source_name = getattr(source, "name", None)
    return source_name if isinstance(source_name, str) and source_name else "metric"


async def run_registered_benchmark_suite(
    dataset_names: Iterable[str] | Iterable[DatasetRunConfig],
    spec: BenchmarkSpec,
    *,
    split: str = "validation",
    profile_kwargs: dict[str, dict[str, object]] | None = None,
    profile_resolver: Callable[..., DatasetProfile] = get_dataset_profile,
    pipeline_runner=run_benchmark,
    client_stats_provider=None,
) -> BenchmarkSuiteResult:
    profile_kwargs = profile_kwargs or {}
    profiles: list[DatasetProfile] = []
    for dataset in dataset_names:
        if isinstance(dataset, DatasetRunConfig):
            name = dataset.name
            resolved_split = dataset.split
            overrides = dict(dataset.profile_kwargs)
            overrides.update(profile_kwargs.get(name, {}))
        else:
            name = str(dataset)
            resolved_split = split
            overrides = dict(profile_kwargs.get(name, {}))
        profiles.append(profile_resolver(
            name, split=resolved_split, **overrides))
    return await run_benchmark_suite(
        tuple(profiles),
        spec,
        metric_resolver=lambda source: source,
        pipeline_runner=pipeline_runner,
        client_stats_provider=client_stats_provider,
    )
