import json
from pathlib import Path

from council_policies.cli import main
from council_policies.models import P2RunResult


def _fake_result(tmp_path: Path) -> P2RunResult:
    return P2RunResult(
        run_id="p2_run_demo",
        output_dir=tmp_path / "p2_run_demo",
        manifest_path=tmp_path / "p2_run_demo" / "manifest.json",
        prediction_file=tmp_path / "p2_run_demo" / "predictions" / "router_dataset__p2_council.jsonl",
        score_file=tmp_path / "p2_run_demo" / "scores" / "router_dataset__p2_council.jsonl",
        summary_files=(tmp_path / "p2_run_demo" / "summaries" / "router_dataset__p2_council.json",),
        aggregate_summary_path=tmp_path / "p2_run_demo" / "suite_metrics.json",
        total_examples=3,
        completed_examples=3,
        failed_examples=0,
        combined_metric=0.8,
        dataset_scores={"musique": 0.8},
    )


def test_cli_delegates_to_p2_runner(monkeypatch, tmp_path: Path, capsys) -> None:
    captured = {}

    async def fake_run_p2_suite(config):
        captured["config"] = config
        return _fake_result(tmp_path)

    monkeypatch.setattr("council_policies.cli.run_p2_suite", fake_run_p2_suite)

    exit_code = main(
        [
            "--output-root",
            str(tmp_path),
            "--dataset",
            "task-aware-llm-council/router_dataset",
            "--provider",
            "openai-compatible",
        ]
    )

    assert exit_code == 0
    assert captured["config"].dataset_name == "task-aware-llm-council/router_dataset"
    assert captured["config"].provider == "openai-compatible"

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "p2_run_demo"
    assert payload["combined_metric"] == 0.8


def test_cli_uses_default_gpu_utilization(monkeypatch, tmp_path: Path, capsys) -> None:
    captured = {}

    async def fake_run_p2_suite(config):
        captured["config"] = config
        return _fake_result(tmp_path)

    monkeypatch.setattr("council_policies.cli.run_p2_suite", fake_run_p2_suite)

    exit_code = main(
        [
            "--output-root",
            str(tmp_path),
            "--provider",
            "vllm",
        ]
    )

    assert exit_code == 0
    assert captured["config"].gpu_utilization == 0.33

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "p2_run_demo"


def test_cli_uses_local_presets(monkeypatch, tmp_path: Path, capsys) -> None:
    captured = {}

    async def fake_run_p2_suite(config):
        captured["config"] = config
        return _fake_result(tmp_path)

    monkeypatch.setattr("council_policies.cli.run_p2_suite", fake_run_p2_suite)

    exit_code = main(
        [
            "--output-root",
            str(tmp_path),
            "--preset",
            "full",
            "--sample-cap",
            "12",
            "--provider",
            "local",
            "--api-base",
            "http://localhost:8000/v1/chat/completions",
        ]
    )

    assert exit_code == 0
    assert captured["config"].max_examples == 12
    assert captured["config"].max_concurrency == 5
    assert captured["config"].provider == "local"
    assert captured["config"].api_base == "http://localhost:8000/v1/chat/completions"

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "p2_run_demo"


def test_cli_allows_row_concurrency_override(monkeypatch, tmp_path: Path, capsys) -> None:
    captured = {}

    async def fake_run_p2_suite(config):
        captured["config"] = config
        return _fake_result(tmp_path)

    monkeypatch.setattr("council_policies.cli.run_p2_suite", fake_run_p2_suite)

    exit_code = main(
        [
            "--output-root",
            str(tmp_path),
            "--max-concurrency",
            "12",
        ]
    )

    assert exit_code == 0
    assert captured["config"].max_concurrency == 12

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "p2_run_demo"
