# Dev Log

## Status

This note summarizes the infrastructure completed so far for the benchmark/council pipeline and the next implementation steps aligned with [baseline-implementation.md](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/docs/baseline-implementation.md).

## Completed So Far

### `llm_gateway` package

The reusable provider layer now exists under [packages/llm_gateway](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/packages/llm_gateway).

Implemented pieces:

- Normalized request/response models:
  - `Message`
  - `PromptRequest`
  - `PromptResponse`
  - `ProviderConfig`
  - `RetryPolicy`
  - `Usage`
- Shared base client abstraction with:
  - request validation
  - async client lifecycle
  - batch generation helper
  - shared retry and backoff handling
  - rate-limit handling with `Retry-After`
  - transport and request/response error normalization
- Provider clients:
  - `OpenAIClient` for direct OpenAI
  - `OpenRouterClient` for OpenRouter
  - `OpenAICompatibleClient` for any OpenAI-style endpoint
- Factory-based construction through `create_client(...)`
- Hugging Face support simplified:
  - no custom Hugging Face client
  - Hugging Face router is now treated as an OpenAI-compatible endpoint
- Public API exposed through [packages/llm_gateway/src/llm_gateway/__init__.py](/home/mbhas/USC/Sem-2/NLP/Task-Aware-LLM-Council/packages/llm_gateway/src/llm_gateway/__init__.py)
- Package README written with:
  - current architecture
  - public API
  - provider usage examples
  - env vars and test commands

### Testing

Implemented and validated:

- Unit compatibility tests for:
  - OpenAI client
  - OpenRouter client
  - OpenAI-compatible flow
  - Hugging Face router via OpenAI-compatible flow
  - factory routing
  - public API exports
- Smoke-test scaffold for:
  - OpenAI
  - OpenRouter
  - Hugging Face router

Known validation state:

- Unit suite is passing
- Hugging Face smoke path has been validated with live API access
- OpenAI and OpenRouter smoke tests are wired but still need live env vars and execution

## Public API

Current public import surface from `llm_gateway`:

- Clients:
  - `OpenAIClient`
  - `OpenRouterClient`
  - `OpenAICompatibleClient`
  - `BaseLLMClient`
- Factory:
  - `create_client`
- Core types:
  - `Message`
  - `PromptRequest`
  - `PromptResponse`
  - `ProviderConfig`
  - `RetryPolicy`
  - `ResponseChoice`
  - `Usage`
- Exceptions:
  - `LLMClientError`
  - `LLMRequestError`
  - `LLMRateLimitError`
  - `LLMResponseError`
  - `LLMTransportError`

## Architectural Decision

The provider layer is intentionally centered around OpenAI-style transport.

- OpenRouter uses a thin wrapper over the OpenAI-compatible client
- OpenAI uses a thin wrapper over the OpenAI-compatible client
- Hugging Face router is treated as another OpenAI-compatible endpoint

This keeps the benchmark/council pipeline generic while avoiding separate client logic for providers that already share the same API shape.

## Next Steps For The Pipeline

These are the next engineering steps after `llm_gateway`, following the baseline guide.

### 1. Central experiment configuration

Create a shared configuration module for:

- model pool
- dataset registry
- output/result paths
- concurrency and timeout settings
- benchmark/council experiment parameters

This should become the single place where model ids, dataset names, and runtime settings are defined.

### 2. Prompt-building layer

Build utilities that convert dataset rows into `PromptRequest` objects.

Required behavior:

- support question-only prompts
- support question + context prompts
- support dataset-specific formatting without leaking provider-specific logic
- keep prompt templates versioned or named so result files can be traced back to prompt format

### 3. Benchmark execution pipeline for P1

Create the single-best-model baseline pipeline that runs each dataset-model pair through `llm_gateway`.

Core requirements:

- iterate over selected models and datasets
- send prompts through `OpenRouterClient` by default
- store normalized raw outputs per example
- track metadata per call:
  - dataset
  - model
  - example id
  - prompt version
  - latency
  - usage if available

### 4. Output persistence

Add a stable output schema for predictions and raw responses.

Recommended artifact split:

- normalized per-example predictions
- raw provider payloads or logs
- aggregated score files

This should map cleanly to the baseline’s `results/` structure for P1 and later P2.

### 5. Dataset-specific evaluation layer

Implement metric functions per dataset type.

Needed from the baseline:

- QA:
  - exact match
  - token F1
- FEVER:
  - label accuracy
- math:
  - numeric accuracy
- code:
  - pass@1 or unit-test-based pass rate

The scorer layer should operate on normalized predictions so it stays independent of the provider client implementation.

### 6. Aggregation and model ranking

After per-example scoring is in place:

- aggregate scores per dataset and per model
- define the primary metric per dataset
- rank models for the P1 baseline
- select the top candidates that later feed into the council policy

### 7. Council pipeline for P2

Once P1 benchmarking exists, build the flat council workflow using the same gateway and prompt layer.

Minimum council loop:

- send the same prompt to the top selected models
- collect independent first responses
- check for majority agreement
- if no majority, send observed responses back for comparison/voting
- persist voting rounds and final selected answer

### 8. Council scoring

Use the same dataset-specific scorers from P1 to score council outputs.

This keeps P1 and P2 results directly comparable.

## Suggested Immediate Order

Recommended next implementation order:

1. configuration module
2. prompt builder
3. benchmark runner
4. result persistence
5. evaluation/scoring
6. aggregation and model ranking
7. council loop
8. council evaluation

## Notes

- For the current repo, `llm_gateway` is the completed foundation layer, not the full benchmark pipeline.
- The next milestone is not another provider client; it is the first working benchmark runner that consumes datasets and produces scored outputs.
