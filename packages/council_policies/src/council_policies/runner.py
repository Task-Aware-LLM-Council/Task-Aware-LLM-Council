from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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
    write_manifest,
)
from model_orchestration.models import OrchestratorConfig

from council_policies.models import PolicyEvaluationResult
from council_policies.runtime import PolicyRuntime


class CouncilBenchmarkRunner:
    """
    Drop-in replacement for run_benchmark (the pipeline_runner in run_benchmark_suite).
    Runs the P2 council policy and writes BenchmarkPrediction JSONL files in the same
    format as P1 so scoring and comparison work unchanged.

    Usage in benchmark_runner CLI:
        runner = CouncilBenchmarkRunner(specialist_config, synthesizer_config)
        await run_registered_benchmark_suite(datasets, spec, pipeline_runner=runner)
    """

    def __init__(
        self,
        specialist_config: OrchestratorConfig,
        synthesizer_config: OrchestratorConfig,
        max_concurrency: int = 10,
    ) -> None:
        self._runtime = PolicyRuntime(specialist_config, synthesizer_config, max_concurrency)

    async def __call__(
        self,
        datasets: Iterable[BenchmarkDataset],
        config: BenchmarkRunConfig,
    ) -> BenchmarkRunResult:
        dataset_list = tuple(datasets)
        run_id = config.run_name or default_run_id()
        run_dir = ensure_run_directory(Path(config.output_root), run_id)
        manifest_path = write_manifest(
            run_dir, run_id=run_id, config=config, datasets=dataset_list
        )
        created_at = datetime.now(timezone.utc).isoformat()

        prediction_files: set[Path] = set()
        total = attempted = completed = failed = skipped = 0

        for dataset in dataset_list:
            for model in config.models:
                path = prediction_path(run_dir, dataset.name, model)
                prediction_files.add(path)

                done_ids = load_recorded_example_ids(path) if config.skip_existing else set()
                to_run = [ex for ex in dataset.examples if ex.example_id not in done_ids]

                total += len(dataset.examples)
                skipped += len(dataset.examples) - len(to_run)
                attempted += len(to_run)

                if not to_run:
                    continue

                results = await self._runtime.run_batch(to_run, dataset_name=dataset.name)

                for result in results:
                    prediction = _result_to_prediction(result, run_id=run_id, model=model, prompt_version=config.prompt_version)
                    append_prediction(path, prediction)
                    if result.status == "success":
                        completed += 1
                    else:
                        failed += 1

        result = BenchmarkRunResult(
            run_id=run_id,
            output_dir=run_dir,
            manifest_path=manifest_path,
            created_at=created_at,
            total_examples=total,
            attempted_examples=attempted,
            completed_examples=completed,
            failed_examples=failed,
            skipped_existing=skipped,
            prediction_files=tuple(sorted(prediction_files)),
        )
        write_manifest(run_dir, run_id=run_id, config=config, datasets=dataset_list, result=result)
        return result


def _result_to_prediction(
    result: PolicyEvaluationResult,
    *,
    run_id: str,
    model: str,
    prompt_version: str,
) -> BenchmarkPrediction:
    metrics = result.metrics
    return BenchmarkPrediction(
        run_id=run_id,
        dataset_name=result.dataset_name,
        model=model,
        example_id=result.example_id,
        status=result.status,
        prompt_version=prompt_version,
        response_text=result.output if result.status == "success" else None,
        latency_ms=metrics.get("latency_ms"),
        usage=metrics.get("token_usage", {}),
        example_metadata={},
        request_metadata={},
        response_metadata=result.metadata,
        error_type=result.error_type,
        error_message=result.error_message,
    )
