import json
from pathlib import Path
from dataclasses import replace
from contextlib import asynccontextmanager

import pytest

from benchmark_runner import (
    BenchmarkCase,
    BenchmarkSpec,
    DatasetRunConfig,
    IterableDatasetSource,
    MetricResult,
    run_registered_benchmark_suite,
    run_benchmark_suite,
)
from benchmarking_pipeline import BenchmarkExample, BenchmarkRunResult
from llm_gateway import Provider, ProviderConfig


class ExactMatchMetric:
    name = "exact_match"

    def score(self, *, case, prediction, dataset_metadata=None):
        answer = case.reference["answer"]
        predicted = prediction.get("response_text", "")
        return MetricResult(
            values={"exact_match": float(predicted == answer)},
            metadata={"dataset": dataset_metadata.get("task") if dataset_metadata else None},
        )


class FailingMetric:
    name = "failing_metric"

    def score(self, *, case, prediction, dataset_metadata=None):
        raise ValueError("metric exploded")


def _spec(tmp_path: Path) -> BenchmarkSpec:
    return BenchmarkSpec(
        provider_config=ProviderConfig(provider=Provider.OPENROUTER),
        models=("model-a", "model-b"),
        output_root=tmp_path,
        suite_name="suite_demo",
        batch_size=2,
        max_concurrency=2,
    )


def _source() -> IterableDatasetSource:
    return IterableDatasetSource(
        name="demo",
        metadata={"task": "qa"},
        cases=[
            BenchmarkCase(
                example=BenchmarkExample(
                    example_id="ex-1",
                    dataset_name="demo",
                    question="Q1",
                ),
                reference={"answer": "Paris"},
            ),
            BenchmarkCase(
                example=BenchmarkExample(
                    example_id="ex-2",
                    dataset_name="demo",
                    question="Q2",
                ),
                reference={"answer": "London"},
            ),
            BenchmarkCase(
                example=BenchmarkExample(
                    example_id="ex-3",
                    dataset_name="demo",
                    question="Q3",
                ),
                reference={"answer": "Rome"},
            ),
        ],
    )


def _second_source() -> IterableDatasetSource:
    return IterableDatasetSource(
        name="demo_two",
        metadata={"task": "qa"},
        cases=[
            BenchmarkCase(
                example=BenchmarkExample(
                    example_id="ex-1",
                    dataset_name="demo_two",
                    question="Q1",
                ),
                reference={"answer": "Paris"},
            ),
            BenchmarkCase(
                example=BenchmarkExample(
                    example_id="ex-2",
                    dataset_name="demo_two",
                    question="Q2",
                ),
                reference={"answer": "London"},
            ),
            BenchmarkCase(
                example=BenchmarkExample(
                    example_id="ex-3",
                    dataset_name="demo_two",
                    question="Q3",
                ),
                reference={"answer": "Rome"},
            ),
        ],
    )


async def fake_pipeline_runner(datasets, config):
    dataset = tuple(datasets)[0]
    run_dir = config.output_root / config.run_name
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions_dir / f"{dataset.name}__{config.models[0]}.jsonl"

    with prediction_path.open("a", encoding="utf-8") as handle:
        for example in dataset.examples:
            response_text = {
                "ex-1": "Paris",
                "ex-2": "Berlin",
                "ex-3": "Rome",
            }[example.example_id]
            payload = {
                "dataset_name": dataset.name,
                "model": config.models[0],
                "example_id": example.example_id,
                "status": "success",
                "prompt_version": config.prompt_version,
                "response_text": response_text,
                "provider": "openrouter",
                "request_metadata": {"example_id": example.example_id},
                "response_metadata": {},
                "usage": {},
            }
            handle.write(json.dumps(payload))
            handle.write("\n")

    return BenchmarkRunResult(
        run_id=config.run_name or "pair_run",
        output_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        created_at="2026-01-01T00:00:00+00:00",
        total_examples=len(dataset.examples),
        attempted_examples=len(dataset.examples),
        completed_examples=len(dataset.examples),
        failed_examples=0,
        skipped_existing=0,
        prediction_files=(prediction_path,),
    )


@pytest.mark.asyncio
async def test_run_benchmark_suite_writes_score_records_and_summaries(tmp_path: Path) -> None:
    result = await run_benchmark_suite(
        (_source(),),
        _spec(tmp_path),
        metric_resolver=lambda _: ExactMatchMetric(),
        pipeline_runner=fake_pipeline_runner,
    )

    assert result.total_pairs == 2
    assert result.total_examples == 6
    assert result.scored_examples == 6
    assert result.failed_examples == 0
    assert len(result.score_files) == 2
    assert len(result.summary_files) == 2
    assert result.aggregate_summary_path.exists()

    score_lines = []
    for path in result.score_files:
        score_lines.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())

    assert len(score_lines) == 6
    assert all(item["status"] == "scored" for item in score_lines)
    assert any(item["metrics"]["exact_match"] == 0.0 for item in score_lines)

    summary = json.loads(result.summary_files[0].read_text(encoding="utf-8"))
    assert summary["metric_name"] == "exact_match"
    assert summary["aggregated_metrics"]["exact_match"] == 2 / 3

    aggregate_rows = json.loads(result.aggregate_summary_path.read_text(encoding="utf-8"))
    assert len(aggregate_rows) == 2
    assert aggregate_rows[0]["primary_metric"] == "exact_match"
    assert aggregate_rows[0]["primary_metric_value"] == 2 / 3


@pytest.mark.asyncio
async def test_run_benchmark_suite_records_metric_failures(tmp_path: Path) -> None:
    result = await run_benchmark_suite(
        (_source(),),
        _spec(tmp_path),
        metric_resolver=lambda _: FailingMetric(),
        pipeline_runner=fake_pipeline_runner,
    )

    assert result.scored_examples == 0
    assert result.failed_examples == 6

    score_records = [
        json.loads(line)
        for path in result.score_files
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["status"] == "metric_error" for record in score_records)


class RegisteredProfile:
    name = "demo"
    primary_metric = "exact_match"
    metadata = {"task": "qa"}

    def iter_cases(self):
        yield from _source().iter_cases()

    def score(self, *, case, prediction, dataset_metadata=None):
        answer = case.reference["answer"]
        predicted = prediction.get("response_text", "")
        return MetricResult(values={"exact_match": float(predicted == answer)})


@pytest.mark.asyncio
async def test_run_registered_benchmark_suite_resolves_profiles(tmp_path: Path) -> None:
    result = await run_registered_benchmark_suite(
        ("demo",),
        _spec(tmp_path),
        profile_resolver=lambda name, **kwargs: RegisteredProfile(),
        pipeline_runner=fake_pipeline_runner,
    )

    assert result.total_pairs == 2
    assert result.scored_examples == 6


@pytest.mark.asyncio
async def test_run_registered_benchmark_suite_accepts_dataset_run_configs(tmp_path: Path) -> None:
    result = await run_registered_benchmark_suite(
        (DatasetRunConfig(name="demo", split="validation"),),
        _spec(tmp_path),
        profile_resolver=lambda name, **kwargs: RegisteredProfile(),
        pipeline_runner=fake_pipeline_runner,
    )

    assert result.total_pairs == 2


@pytest.mark.asyncio
async def test_run_benchmark_suite_reuses_one_local_container_per_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    lifecycle_events: list[str] = []
    pipeline_calls: list[tuple[str, str, str | None]] = []

    @asynccontextmanager
    async def fake_managed_local_provider_config(config, *, model):
        lifecycle_events.append(f"start:{model}")
        yield replace(config, api_base=f"http://127.0.0.1:80/{model}")
        lifecycle_events.append(f"stop:{model}")

    async def fake_local_pipeline_runner(datasets, config):
        dataset = tuple(datasets)[0]
        pipeline_calls.append((config.models[0], dataset.name, config.provider_config.api_base))
        return await fake_pipeline_runner(datasets, config)

    monkeypatch.setattr(
        "benchmark_runner.suite.managed_local_provider_config",
        fake_managed_local_provider_config,
    )

    spec = BenchmarkSpec(
        provider_config=ProviderConfig(
            provider=Provider.LOCAL,
            default_model="model-a",
            default_params={"local_launch_image": "vllm-openai.sif"},
        ),
        models=("model-a", "model-b"),
        output_root=tmp_path,
        suite_name="suite_demo",
        batch_size=2,
        max_concurrency=2,
    )

    result = await run_benchmark_suite(
        (_source(), _second_source()),
        spec,
        metric_resolver=lambda _: ExactMatchMetric(),
        pipeline_runner=fake_local_pipeline_runner,
    )

    assert result.total_pairs == 4
    assert lifecycle_events == [
        "start:model-a",
        "stop:model-a",
        "start:model-b",
        "stop:model-b",
    ]
    assert pipeline_calls == [
        ("model-a", "demo", "http://127.0.0.1:80/model-a"),
        ("model-a", "demo", "http://127.0.0.1:80/model-a"),
        ("model-a", "demo_two", "http://127.0.0.1:80/model-a"),
        ("model-a", "demo_two", "http://127.0.0.1:80/model-a"),
        ("model-b", "demo", "http://127.0.0.1:80/model-b"),
        ("model-b", "demo", "http://127.0.0.1:80/model-b"),
        ("model-b", "demo_two", "http://127.0.0.1:80/model-b"),
        ("model-b", "demo_two", "http://127.0.0.1:80/model-b"),
    ]
