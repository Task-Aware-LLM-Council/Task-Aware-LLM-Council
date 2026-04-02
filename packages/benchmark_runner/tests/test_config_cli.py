import json
from pathlib import Path

from benchmark_runner.cli import main
from benchmark_runner.config import DATASET_CONFIGS, MODEL_POOL, get_dataset_configs, get_preset_spec
from benchmark_runner.models import BenchmarkSuiteResult


def test_get_preset_spec_uses_expected_sample_caps(tmp_path: Path) -> None:
    pilot = get_preset_spec("pilot", output_root=tmp_path)
    full = get_preset_spec("full", output_root=tmp_path)

    assert pilot.max_examples_per_dataset == 50
    assert full.max_examples_per_dataset == 160
    assert pilot.models == MODEL_POOL


def test_get_dataset_configs_filters_by_name() -> None:
    selected = get_dataset_configs(("musique", "fever"))
    assert tuple(config.name for config in selected) == ("musique", "fever")
    assert len(DATASET_CONFIGS) >= len(selected)


def test_cli_runs_registered_suite_and_prints_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    captured = {}

    async def fake_run_registered(dataset_names, spec, **kwargs):
        captured["dataset_names"] = tuple(dataset_names)
        captured["spec"] = spec
        return BenchmarkSuiteResult(
            suite_id="suite_demo",
            output_dir=tmp_path / "suite_demo",
            manifest_path=tmp_path / "suite_demo" / "manifest.json",
            score_files=(),
            summary_files=(),
            aggregate_summary_path=tmp_path / "suite_demo" / "suite_metrics.json",
            total_pairs=2,
            total_examples=10,
            scored_examples=10,
            failed_examples=0,
        )

    monkeypatch.setattr("benchmark_runner.cli.run_registered_benchmark_suite", fake_run_registered)

    exit_code = main(
        [
            "--preset",
            "pilot",
            "--output-root",
            str(tmp_path),
            "--datasets",
            "musique",
            "fever",
            "--models",
            "model-a",
        ]
    )

    assert exit_code == 0
    assert [config.name for config in captured["dataset_names"]] == ["musique", "fever"]
    assert captured["spec"].models == ("model-a",)

    payload = json.loads(capsys.readouterr().out)
    assert payload["suite_id"] == "suite_demo"
    assert payload["total_pairs"] == 2
