# benchmark_runner

`benchmark_runner` is the experiment-facing package for running benchmark
model-dataset pairs end to end. It sits above `benchmarking_pipeline` and
`task_eval`.

Responsibilities:

- choose models and datasets for a run
- launch dataset-model pairs through `benchmarking_pipeline`
- use `task_eval` dataset profiles for streaming cases and scoring
- write per-example scores, per-pair summaries, and one flattened suite summary

## Package Role

Current stack:

- `llm_gateway`: provider communication and shared local vLLM runtime management
- `benchmarking_pipeline`: prompt execution and prediction persistence
- `task_eval`: dataset profiles, normalization, extraction, and scoring
- `benchmark_runner`: preset/config handling, suite execution, and aggregate summaries

## Public API

- `BenchmarkSpec`
- `DatasetRunConfig`
- `AggregateMetricRow`
- `BenchmarkSuiteResult`
- `build_benchmark_spec(...)`
- `get_preset_spec(...)`
- `get_dataset_configs(...)`
- `run_benchmark_suite(...)`
- `run_registered_benchmark_suite(...)`
- `main(...)`

## Recommended Entry Point

Use the registered suite runner when you want the built-in dataset profiles from
`task_eval`:

```python
from pathlib import Path

from benchmark_runner import get_dataset_configs, get_preset_spec, run_registered_benchmark_suite


spec = get_preset_spec("pilot", output_root=Path("results/benchmark_suite"))
datasets = get_dataset_configs()

result = await run_registered_benchmark_suite(datasets, spec)
print(result.aggregate_summary_path)
```

## CLI

The package now exposes a CLI:

```bash
uv run benchmark-runner --preset pilot --provider vllm (other providers are not tested)
```

Useful flags:

- `--preset pilot|full`
- `--output-root <path>`
- `--datasets musique fever`
- `--models qwen/qwen3-32b deepseek/deepseek-r1`
- `--sample-cap 20`
- `--split validation`

The CLI prints a small JSON summary containing:

- `suite_id`
- `output_dir`
- `aggregate_summary_path`
- `total_pairs`
- `total_examples`
- `scored_examples`
- `failed_examples`

## Output Layout

Each suite writes to:

```text
<output_root>/<suite_id>/
```

Artifacts:

- `manifest.json`
- `predictions/...` from `benchmarking_pipeline`
- `scores/<dataset>__<model>.jsonl`
- `summaries/<dataset>__<model>.json`
- `suite_metrics.json`

`suite_metrics.json` is the easiest file to use for model comparison because it
contains one flattened row per dataset-model pair with:

- `dataset_name`
- `model`
- `primary_metric`
- `primary_metric_value`
- `aggregated_metrics`
- `scored_examples`
- `failed_examples`

## Presets

Current presets:

- `pilot`: 50 examples per dataset
- `full`: 160 examples per dataset

Model defaults come from the baseline model pool in the package config.

Local vLLM startup is now delegated to `llm_gateway`, so benchmark runner only
passes `provider=local` plus any `local_launch_*` runtime metadata through to
the shared runtime layer.

## Validation

Current benchmark runner coverage includes:

- chunked case handling
- suite orchestration
- aggregate summary generation
- registered dataset-profile execution
- config and CLI behavior
