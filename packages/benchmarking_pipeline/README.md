# benchmarking_pipeline

`benchmarking_pipeline` is the orchestration layer for benchmark runs. It takes
normalized benchmark examples, converts them into `llm_gateway.PromptRequest`
objects, executes dataset-model runs, and writes reproducible prediction
artifacts plus run metadata.

## Purpose

This package is the execution layer for the P1 baseline workflow.

It is responsible for:

- taking normalized benchmark examples as input
- calling LLM providers through `llm_gateway`
- managing dataset-model execution loops
- persisting prediction artifacts in a stable format
- supporting interrupted-run resume through skip-existing behavior

It is intentionally not responsible for:

- dataset loading
- dataset normalization from raw source rows
- metric calculation
- score aggregation
- model ranking
- council logic

## How It Fits With `llm_gateway`

`benchmarking_pipeline` depends directly on `llm_gateway` rather than defining
its own provider abstraction.

It uses:

- `ProviderConfig` for provider settings
- `PromptRequest` as the request format sent to the gateway
- `create_client(...)` or an injected client instance for execution

This keeps provider logic in one place and keeps the pipeline package focused on
orchestration and persistence.

## Public API

Import from the package root:

```python
from benchmarking_pipeline import (
    BenchmarkDataset,
    BenchmarkExample,
    BenchmarkRunConfig,
    run_benchmark,
)
```

Public symbols:

- `BenchmarkExample`
- `BenchmarkDataset`
- `BenchmarkRunConfig`
- `BenchmarkPrediction`
- `BenchmarkRunResult`
- `build_prompt_request`
- `run_benchmark`

### `BenchmarkExample`

One normalized example to be sent to a model.

Fields:

- `example_id`
- `dataset_name`
- `question`
- `context`
- `system_prompt`
- `messages`
- `metadata`

Use either:

- `question` / `context` / `system_prompt`

or:

- explicit `messages`

### `BenchmarkDataset`

A named collection of normalized examples plus optional dataset-level metadata.

### `BenchmarkRunConfig`

Run-level settings for the pipeline.

Important fields:

- `provider_config`
- `models`
- `output_root`
- `run_name`
- `prompt_version`
- `max_concurrency`
- `continue_on_error`
- `skip_existing`
- `save_raw_response`
- `temperature`
- `max_tokens`
- `stop_sequences`
- `provider_params`

### `run_benchmark(...)`

Async entrypoint that:

- iterates over datasets and models
- builds prompts
- calls the LLM client
- writes prediction artifacts
- returns a `BenchmarkRunResult`

You can either:

- let it create the client from `provider_config`

or:

- pass an already-created `llm_gateway` client for tighter control in tests or
  custom runtime setups

## Input Requirements

This package expects examples to already be normalized.

That means some other layer should prepare:

- stable `example_id`
- correct `dataset_name`
- question text
- optional context
- optional system prompt
- optional prebuilt message list
- any dataset metadata you want to persist with the prediction

It also expects provider credentials and endpoint settings to be handled through
`llm_gateway.ProviderConfig`.

## Prompt Construction

`build_prompt_request(...)` converts a `BenchmarkExample` into a
`PromptRequest`.

Current behavior:

- if `messages` is present, it uses those as the source of truth
- otherwise it builds a request from `system_prompt`, `question`, and `context`
- it attaches trace metadata:
  - `dataset_name`
  - `example_id`
  - `prompt_version`
- it applies run-level generation settings from `BenchmarkRunConfig`

## Execution Behavior

For each dataset and each model, the runner:

1. resolves the output file path
2. loads already recorded `example_id`s when `skip_existing=True`
3. builds one prompt per remaining example
4. executes requests with bounded async concurrency
5. writes each prediction record to disk as soon as it completes
6. records failures as structured error rows when `continue_on_error=True`
7. writes a manifest before and after the run

This means partial progress is preserved even if a longer run is interrupted.

## Output Layout

Each run writes to:

```text
<output_root>/<run_id>/
```

Current artifacts:

- `manifest.json`
- `predictions/<dataset>__<model>.jsonl`

### `manifest.json`

Contains:

- `run_id`
- serialized run config
- dataset inventory
- final run summary

### Prediction JSONL rows

Each row is a normalized `BenchmarkPrediction`.

Success records include:

- dataset
- model
- example id
- prompt version
- response text
- latency
- request id
- finish reason
- provider
- usage
- request metadata
- response metadata

Failure records include:

- dataset
- model
- example id
- prompt version
- request metadata
- `error_type`
- `error_message`

If `save_raw_response=True`, raw provider payloads are also saved.

## Resume / Skip Behavior

If `skip_existing=True`, the runner reads existing JSONL files for a given
dataset-model pair and skips any example whose `example_id` is already present.

This is intended for:

- interrupted long runs
- reruns with the same named output directory
- incremental continuation without paying again for finished examples

## Minimal Example

```python
from pathlib import Path

from benchmarking_pipeline import (
    BenchmarkDataset,
    BenchmarkExample,
    BenchmarkRunConfig,
    run_benchmark,
)
from llm_gateway import Provider, ProviderConfig


dataset = BenchmarkDataset(
    name="demo",
    examples=(
        BenchmarkExample(
            example_id="1",
            dataset_name="demo",
            question="What is the capital of France?",
        ),
    ),
)

config = BenchmarkRunConfig(
    provider_config=ProviderConfig(
        provider=Provider.OPENROUTER,
        default_model="openai/gpt-4o-mini",
    ),
    models=("openai/gpt-4o-mini",),
    output_root=Path("results/benchmarking"),
    run_name="demo_run",
    prompt_version="v1",
    max_concurrency=5,
)

result = await run_benchmark((dataset,), config)
print(result.output_dir)
```

## Current Validation

Current unit coverage includes:

- prompt construction
- manifest and prediction persistence
- partial-failure handling
- resume behavior

Validated command:

```bash
uv run pytest packages/benchmarking_pipeline/tests -q
```

Current result:

```text
5 passed
```

## Current Limitations

Still intentionally out of scope:

- dataset adapters and raw dataset loading
- metric calculation
- score aggregation and ranking
- council execution
- CLI workflow
- reporting and analysis helpers

These are the next layers that should be built on top of the current pipeline
rather than folded back into it.
