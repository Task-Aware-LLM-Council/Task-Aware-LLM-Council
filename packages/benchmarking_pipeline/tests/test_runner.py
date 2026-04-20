import json
from pathlib import Path

import pytest

from benchmarking_pipeline import (
    BenchmarkDataset,
    BenchmarkExample,
    BenchmarkRunConfig,
    run_benchmark,
)
from llm_gateway import LLMTransportError, PromptResponse, Provider, ProviderConfig, Usage


class FakeGatewayClient:
    def __init__(self, fail_example_ids: set[str] | None = None) -> None:
        self.fail_example_ids = fail_example_ids or set()
        self.requests: list[object] = []
        self.closed = False

    async def generate(self, request):
        self.requests.append(request)
        example_id = request.metadata["example_id"]
        if example_id in self.fail_example_ids:
            raise LLMTransportError(f"temporary failure for {example_id}")

        return PromptResponse(
            model=request.model or "unknown-model",
            text=f"answer:{example_id}",
            latency_ms=12.5,
            request_id=f"req-{example_id}",
            finish_reason="stop",
            provider="openrouter",
            usage=Usage(input_tokens=5, output_tokens=3, total_tokens=8),
            metadata={"attempt_count": 1},
            raw_response={"id": f"raw-{example_id}"},
        )

    async def close(self) -> None:
        self.closed = True


def _dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        name="demo",
        examples=(
            BenchmarkExample(
                example_id="ex-1",
                dataset_name="demo",
                question="Question 1?",
                metadata={"split": "pilot"},
            ),
            BenchmarkExample(
                example_id="ex-2",
                dataset_name="demo",
                question="Question 2?",
            ),
        ),
    )


def _config(tmp_path: Path) -> BenchmarkRunConfig:
    return BenchmarkRunConfig(
        provider_config=ProviderConfig(provider=Provider.OPENROUTER),
        models=("model-a", "model-b"),
        output_root=tmp_path,
        run_name="benchmark_run",
        prompt_version="v2",
        max_concurrency=2,
        save_raw_response=True,
    )


@pytest.mark.asyncio
async def test_run_benchmark_writes_manifest_and_predictions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dataset = _dataset()
    client = FakeGatewayClient()

    result = await run_benchmark((dataset,), config, client=client)

    assert result.run_id == "benchmark_run"
    assert result.total_examples == 4
    assert result.attempted_examples == 4
    assert result.completed_examples == 4
    assert result.failed_examples == 0
    assert result.skipped_existing == 0
    assert len(result.prediction_files) == 2

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "benchmark_run"
    assert manifest["config"]["prompt_version"] == "v2"
    assert manifest["datasets"][0]["name"] == "demo"
    assert manifest["summary"]["completed_examples"] == 4

    prediction_lines = []
    for path in result.prediction_files:
        assert path.exists()
        prediction_lines.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )

    assert len(prediction_lines) == 4
    assert all(item["status"] == "success" for item in prediction_lines)
    assert prediction_lines[0]["raw_response"]["id"].startswith("raw-")
    assert prediction_lines[0]["request_metadata"]["prompt_version"] == "v2"
    assert client.closed is False


@pytest.mark.asyncio
async def test_run_benchmark_records_partial_failures(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = FakeGatewayClient(fail_example_ids={"ex-2"})

    result = await run_benchmark((_dataset(),), config, client=client)

    assert result.attempted_examples == 4
    assert result.completed_examples == 2
    assert result.failed_examples == 2

    prediction_lines = []
    for path in result.prediction_files:
        prediction_lines.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )

    error_records = [item for item in prediction_lines if item["status"] == "error"]
    assert len(error_records) == 2
    assert all(item["error_type"] == "LLMTransportError" for item in error_records)


@pytest.mark.asyncio
async def test_run_benchmark_skips_existing_predictions_on_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_client = FakeGatewayClient()
    second_client = FakeGatewayClient()
    dataset = _dataset()

    first_result = await run_benchmark((dataset,), config, client=first_client)
    second_result = await run_benchmark((dataset,), config, client=second_client)

    assert first_result.completed_examples == 4
    assert second_result.attempted_examples == 0
    assert second_result.completed_examples == 0
    assert second_result.skipped_existing == 4
    assert len(second_client.requests) == 0
