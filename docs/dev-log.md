# Dev Log

## Status

This note tracks the benchmark/council infrastructure completed so far and the
next implementation steps aligned with
[baseline-implementation.md](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/docs/baseline-implementation.md).

## Completed So Far

### `llm_gateway`

The reusable provider layer exists under
[packages/llm_gateway](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/packages/llm_gateway).

Implemented pieces:

- normalized request/response models:
  - `Message`
  - `PromptRequest`
  - `PromptResponse`
  - `ProviderConfig`
  - `RetryPolicy`
  - `Usage`
- shared base client behavior:
  - request validation
  - async client lifecycle
  - batch generation helper
  - shared retry/backoff handling
  - rate-limit handling with `Retry-After`
  - transport and request/response error normalization
- provider clients:
  - `OpenAIClient`
  - `OpenRouterClient`
  - `OpenAICompatibleClient`
- factory-based construction through `create_client(...)`
- Hugging Face router support through the OpenAI-compatible path
- package-level public API from
  [packages/llm_gateway/src/llm_gateway/__init__.py](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/packages/llm_gateway/src/llm_gateway/__init__.py)

Validation state:

- unit compatibility tests are passing
- Hugging Face live smoke path has been validated
- OpenAI and OpenRouter smoke tests are scaffolded and ready for live credentials

### `benchmarking_pipeline`

The benchmark execution layer exists under
[packages/benchmarking_pipeline](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/packages/benchmarking_pipeline).

Implemented pieces:

- normalized pipeline types:
  - `BenchmarkExample`
  - `BenchmarkDataset`
  - `BenchmarkRunConfig`
  - `BenchmarkPrediction`
  - `BenchmarkRunResult`
- prompt construction from normalized examples into `llm_gateway.PromptRequest`
- async dataset-model execution through `llm_gateway`
- deterministic output persistence:
  - `manifest.json`
  - per-dataset/per-model JSONL prediction files
- per-example metadata capture:
  - dataset
  - model
  - example id
  - prompt version
  - latency
  - provider metadata
  - usage when available
- structured error recording when individual calls fail
- resume behavior by skipping already recorded example ids
- package-level public API from
  [packages/benchmarking_pipeline/src/benchmarking_pipeline/__init__.py](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/packages/benchmarking_pipeline/src/benchmarking_pipeline/__init__.py)

Validation state:

- unit tests cover prompt construction, persistence, partial failures, and resume behavior
- validated command:
  - `uv run pytest packages/benchmarking_pipeline/tests -q`
- current result:
  - `5 passed`

### `task_eval`

The reusable dataset-profile and evaluation package now exists under
[packages/task_eval](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/packages/task_eval).

Implemented pieces:

- shared evaluation types:
  - `EvaluationCase`
  - `MetricResult`
- shared interfaces:
  - `DatasetProfile`
  - `MetricCalculator`
- reusable normalization helpers:
  - SQuAD-style answer normalization
  - tokenization
  - FEVER label normalization
- reusable extraction helpers:
  - QA answer extraction
  - MCQ letter extraction
  - FEVER label extraction
  - math answer extraction
  - code extraction
- reusable scoring helpers:
  - exact match
  - token F1
  - label accuracy
  - numeric accuracy
  - pass@1
- dataset profiles for:
  - `musique`
  - `quality`
  - `fever`
  - `hardmath`
  - `humaneval_plus`
- dataset profile registry through `get_dataset_profile(...)`

Validation state:

- helper and profile tests are passing
- validated together with benchmark runner integration:
  - `uv run pytest packages/task_eval/tests packages/benchmark_runner/tests -q`
- current combined result:
  - `12 passed`

### `benchmark_runner`

The top-level benchmark suite and launch package exists under
[packages/benchmark_runner](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/packages/benchmark_runner).

Implemented pieces:

- suite-level orchestration types:
  - `BenchmarkCase`
  - `BenchmarkSpec`
  - `DatasetRunConfig`
  - `ScoreRecord`
  - `ScoreSummary`
  - `AggregateMetricRow`
  - `BenchmarkSuiteResult`
- chunked, memory-safe dataset-model execution over streamed cases
- reuse of `benchmarking_pipeline` for prediction generation
- use of `task_eval` dataset profiles as both:
  - case sources
  - metric calculators
- score artifact persistence:
  - per-example score JSONL
  - per-dataset/per-model summary JSON
  - flattened suite summary JSON
- registered benchmark execution through `run_registered_benchmark_suite(...)`
- benchmark presets and config:
  - model pool
  - dataset configs
  - `pilot` preset
  - `full` preset
- CLI entrypoint:
  - `uv run benchmark-runner --preset pilot`

Current output shape per suite:

- `manifest.json`
- `predictions/...` via `benchmarking_pipeline`
- `scores/<dataset>__<model>.jsonl`
- `summaries/<dataset>__<model>.json`
- `suite_metrics.json`

Validation state:

- benchmark runner tests cover:
  - chunked case handling
  - suite orchestration
  - metric-failure handling
  - registered dataset-profile execution
  - config and CLI behavior
  - aggregate summary generation
- validated command:
  - `uv run pytest packages/benchmark_runner/tests -q`
- current result:
  - `8 passed`

## Architectural State

The repo now has four benchmark layers in place:

- `llm_gateway` handles provider communication
- `benchmarking_pipeline` handles prediction generation and prediction artifacts
- `task_eval` handles dataset-specific normalization, extraction, and metrics
- `benchmark_runner` handles experiment presets, dataset-model pair execution, and score artifacts

This means the infrastructure for running per-dataset metrics per model now
exists. The main remaining work is no longer package creation. The main
remaining work is hardening the dataset/profile configs and running real
benchmark suites successfully against provider APIs.

## What Is Runnable Now

The intended end-to-end path is:

1. configure model list and dataset list in
   [packages/benchmark_runner/src/benchmark_runner/config.py](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/packages/benchmark_runner/src/benchmark_runner/config.py)
2. ensure provider credentials are available in the shell environment
3. run:
   - `uv run benchmark-runner --preset pilot`
4. inspect:
   - `scores/*.jsonl`
   - `summaries/*.json`
   - `suite_metrics.json`

The flattened `suite_metrics.json` artifact is the easiest file to use for
model comparison because it contains one row per dataset-model pair with:

- `dataset_name`
- `model`
- `primary_metric`
- `primary_metric_value`
- `aggregated_metrics`
- `scored_examples`
- `failed_examples`

## Current Gaps / Pending Work

### 1. Provider configuration and auth validation

The runner currently works structurally, but real benchmark runs require valid
provider credentials in the shell environment.

Current observed issue from the first real run:

- OpenRouter requests failed with `401`
- error message:
  - `No cookie auth credentials found`

That means the immediate runtime blocker is provider auth/config, not the
pipeline code.

## Latest Architectural Decision

To avoid hosted API rate limits during first-submission benchmarking, local GPU
inference is now routed through
[packages/llm_gateway](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/packages/llm_gateway)
instead of introducing a separate package boundary for execution.

Decision details:

- local inference is treated as a first-class `llm_gateway` provider
- the active local runtime target is an external OpenAI-compatible vLLM server
  managed outside Python, typically via Apptainer on CARC
- `benchmark_runner` can now optionally start and stop Apptainer-backed local
  servers per model using config-driven launch parameters
- `llm_gateway` only consumes the endpoint URL and does not manage the container
  lifecycle itself
- short-lived local clients fit one-server-per-port execution and future
  multi-instance council orchestration
- this preserves one inference surface for:
  - single-model benchmarking now
  - council inference later

This keeps `benchmarking_pipeline` and `benchmark_runner` unchanged at the
request/response contract level while moving container/runtime lifecycle
management outside `llm_gateway`.

### 2. Final model pool and dataset selection

The currently active benchmark config was temporarily narrowed to one model and
one dataset for validation.

Pending work:

- finalize the full model pool
- finalize the active dataset list
- verify model ids match the target provider exactly

### 3. Dataset profile hardening

`task_eval` now has real profile classes, but each profile still needs final
schema verification against the real dataset source.

Pending checks per dataset:

- exact Hugging Face dataset id
- exact split names
- exact row field names
- reference field extraction correctness
- prompt/context shaping correctness

This is especially important for:

- `quality`
- `fever`
- `hardmath`
- `humaneval_plus`

### 4. Humaneval+ execution policy

`pass_at_1` support exists, but code-evaluation behavior still needs practical
verification under the intended runtime environment.

Pending work:

- verify test harness field mapping
- verify timeout settings
- verify sandbox/compute-node execution policy for real runs

### 5. First successful pilot benchmark

After auth and profile configs are verified, the next milestone is:

- run a successful pilot benchmark across the selected model/dataset set
- confirm `suite_metrics.json` contains real scored outputs instead of only failures

### 6. Aggregation and model ranking

After successful pilot runs:

- aggregate results by dataset and model
- define primary metric reporting per dataset
- rank models for the P1 baseline
- select top candidates for later council experiments

### 7. Council pipeline and scoring

After the P1 path is stable:

- build the flat council loop
- reuse the same prompt and scoring layers
- persist council rounds and final outputs

## Suggested Immediate Order

Recommended next implementation / validation order:

1. fix provider auth and provider config in the runtime environment
2. finalize model pool and dataset selection in `benchmark_runner/config.py`
3. verify each `task_eval` dataset profile against the real dataset schema
4. run a successful pilot benchmark
5. inspect `suite_metrics.json` and pair summaries
6. add model-ranking / aggregation logic
7. move on to council execution

## Notes

- `benchmarking_pipeline` is intentionally prediction-only; it does not own dataset loading or scoring.
- `task_eval` is intentionally reusable; dataset-specific logic should continue to live there rather than drifting into `benchmark_runner`.
- `benchmark_runner` is now the correct package to use for end-to-end benchmark execution.
- The current repo is close to first real benchmark results; the biggest remaining issues are configuration accuracy and runtime validation, not missing infrastructure.
