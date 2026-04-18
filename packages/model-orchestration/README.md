# model-orchestration

`model-orchestration` is a small library layer that gives council-style code a
stable way to talk to a few named model roles without embedding provider or
runtime wiring in the caller.

It sits above `llm_gateway` and is responsible for:

- mapping semantic roles like `qa`, `math`, or `general` to concrete models
- lazily creating and managing one client per configured role
- supporting both HTTP-backed providers and local-runtime resolution
- returning structured response metadata for downstream metrics/logging
- optionally recording request/response events

It does not implement council voting, aggregation, routing policy, or scoring.

For local vLLM usage, the package now owns the default launch configuration for
the three canonical roles instead of requiring callers to hand-build raw
`ProviderConfig.default_params`.

## Installation / Environment

This package is intended to be used from the workspace:

```bash
uv run python
```

The real public surface is the Python API, not the CLI script. The package
script only prints a library-use message.

For local-runtime usage, prefer the built-in local vLLM helper described below.
It applies the same effective launch settings currently used in
`benchmark_runner/cli.py` and assigns deterministic ports per canonical role.

## Main Exports

Primary public symbols:

- `ModelOrchestrator`
- `ModelSpec`
- `OrchestratorConfig`
- `OrchestratorRequest`
- `OrchestratorResponse`
- `OrchestratorCallRecord`
- `LocalVLLMPresetConfig`
- `JSONLRecordingConfig`
- `JSONLRecorder`
- `InMemoryRecorder`
- `NoOpRecorder`
- `build_default_orchestrator_config(...)`
- `build_default_local_vllm_orchestrator_config(...)`

Canonical request type:

- `llm_gateway.PromptRequest`

## Quick Start

### Default 3-role orchestrator

```python
from llm_gateway import PromptRequest, Provider
from model_orchestration import ModelOrchestrator, build_default_orchestrator_config

config = build_default_orchestrator_config(
    provider=Provider.OPENAI_COMPATIBLE,
    api_base="http://127.0.0.1:8000/v1/chat/completions",
)

async with ModelOrchestrator(config) as orchestrator:
    response = await orchestrator.qa_client.get_response(
        PromptRequest(user_prompt="Answer briefly: What is the capital of France?")
    )
    print(response.text)
    print(response.model)
    print(response.provider)
```

### Generic alias-based access

```python
from llm_gateway import PromptRequest

client = orchestrator.get_client("math")
response = await client.get_response(
    PromptRequest(user_prompt="Solve: 17 * 19")
)
```

### Direct orchestrator call

```python
from llm_gateway import PromptRequest

response = await orchestrator.run(
    PromptRequest(user_prompt="Classify this claim."),
    target="general",
)
```

### Sync wrapper

```python
from model_orchestration import ModelOrchestrator, build_default_orchestrator_config

config = build_default_orchestrator_config()
orchestrator = ModelOrchestrator(config)

response = orchestrator.qa_client.get_response_sync(
    user_prompt="Summarize this in one line."
)
print(response.text)
```

### Default local vLLM orchestrator

```python
from model_orchestration import (
    ModelOrchestrator,
    build_default_local_vllm_orchestrator_config,
)

config = build_default_local_vllm_orchestrator_config()

async with ModelOrchestrator(config) as orchestrator:
    qa = await orchestrator.qa_client.get_response(user_prompt="Answer this.")
    math = await orchestrator.math_client.get_response(user_prompt="Solve this.")
    general = await orchestrator.general_client.get_response(user_prompt="Check this claim.")
```

This creates one local vLLM-backed client per canonical role:

- `qa` on port `8000`
- `reasoning` on port `8001`
- `general` on port `8002`
- `synthesizer` on port `8003`

Aliases reuse the canonical client:

- `math` and `code` reuse `reasoning`
- `fever` reuses `general`

## Default Role Mapping

`build_default_orchestrator_config(...)` creates four canonical roles:

- `qa` -> `google/gemma-2-9b-it`
- `reasoning` -> `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `general` -> `Qwen/Qwen2.5-14B-Instruct`
- `synthesizer` -> `Qwen/Qwen2.5-32B-Instruct`

Default aliases:

- `qa`
- `reasoning`, `math`, `code`
- `general`, `fever`
- `synthesizer`

The default role is `general`. The `synthesizer` role is the P3/P4 fan-out
referee — it is invoked only by `council_policies.synthesis.synthesize()`
and is intentionally excluded from `council_policies.models.TASK_TO_ROLE`,
so rule-based routing can never dispatch a user task to it.

When local vLLM mode is used, the package assigns deterministic ports by role:

- `qa` -> `base_port + 0`
- `reasoning` -> `base_port + 1`
- `general` -> `base_port + 2`
- `synthesizer` -> `base_port + 3`

> **Memory note.** The default local preset assumes all four roles can
> co-reside (`gpu_memory_utilization=0.33` per role). On single-GPU setups
> where that budget is too tight, use `ModelOrchestrator.fanout([...])` —
> see [FanoutSession](#fanoutsession-swap-safe-multi-role-dispatch) — to
> serialize loads and raise `gpu_memory_utilization` in the preset.

You can override the concrete model ids without changing caller code:

```python
config = build_default_orchestrator_config(
    qa_model="Qwen/Qwen2.5-7B-Instruct",
    reasoning_model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    general_model="google/gemma-2-9b-it",
    synthesizer_model="Qwen/Qwen2.5-14B-Instruct",
)
```

## Local vLLM Preset

Use `build_default_local_vllm_orchestrator_config(...)` when you want the
package to own local vLLM startup settings.

The built-in defaults match the current effective vLLM settings used in
`benchmark_runner/cli.py`:

- `image="vllm-openai_latest.sif"`
- `bind="/scratch1/$USER/.cache"`
- `startup_timeout_seconds=600`
- `max_model_len="8192"`
- `gpu_memory_utilization=0.33`
- `quantization="compressed-tensors"`
- `base_port=8000`

You can override them through `LocalVLLMPresetConfig`:

```python
from model_orchestration import (
    LocalVLLMPresetConfig,
    build_default_local_vllm_orchestrator_config,
)

config = build_default_local_vllm_orchestrator_config(
    preset=LocalVLLMPresetConfig(
        base_port=8100,
        gpu_memory_utilization=0.4,
    )
)
```

You can also apply global and per-role launch overrides:

```python
config = build_default_local_vllm_orchestrator_config(
    preset=LocalVLLMPresetConfig(
        base_port=8100,
        provider_defaults={"temperature": 0.1},
        role_overrides={"reasoning": {"local_launch_port": 8125}},
    )
)
```

## API Reference

### `ModelSpec`

Defines one canonical model role.

Fields:

- `role`: canonical role name
- `model`: concrete model id used when the request does not specify one
- `provider_config`: `llm_gateway.ProviderConfig` for this role
- `aliases`: alternate names accepted by `get_client(...)` and `run(..., target=...)`
- `description`: optional human-readable description
- `request_defaults`: default request values merged into `PromptRequest`

`request_defaults` currently supports the fields merged by the orchestrator:

- `metadata`
- `provider_params`
- `temperature`
- `max_tokens`
- `system_prompt`
- `user_prompt`
- `context`
- `stop_sequences`

### `OrchestratorConfig`

Top-level orchestrator configuration.

Fields:

- `models`: tuple of `ModelSpec`
- `default_role`: role used when the caller does not pass a target
- `recording`: optional `JSONLRecordingConfig`
- `mode_label`: optional override for reported mode, otherwise inferred as `local` or `http`

### `LocalVLLMPresetConfig`

Local vLLM preset used by `build_default_local_vllm_orchestrator_config(...)`.

Important fields:

- `base_port`
- `image`
- `bind`
- `startup_timeout_seconds`
- `max_model_len`
- `gpu_memory_utilization`
- `quantization`
- optional launch overrides such as:
  - `dtype`
  - `load_format`
  - `env`
  - `extra_args`
  - `cpu_offload_gb`
- `provider_defaults` for global launch/request defaults
- `role_overrides` for per-role launch overrides

### `ModelOrchestrator`

Main entrypoint.

Important methods and attributes:

- `get_client(alias)`
- `run(request, target=...)`
- `run_sync(request=None, target=..., ...)`
- `fanout([roles...])` — see below
- `build_prompt_request(...)`
- `close()`
- `qa_client`
- `reasoning_client`
- `math_client`
- `code_client`
- `general_client`
- `fever_client`

Use it as an async context manager when possible:

```python
async with ModelOrchestrator(config) as orchestrator:
    ...
```

### FanoutSession: swap-safe multi-role dispatch

`orchestrator.fanout([...])` returns an async context manager that scopes a
multi-role call sequence. Within the session, only one role's weights are
resident at a time — the session swaps on role change by tearing down the
previous role's client (including its vLLM container lifecycle) before
opening the next.

```python
async with orchestrator.fanout(["qa", "reasoning", "synthesizer"]) as session:
    qa_responses       = await session.run_role("qa", [request])
    reasoning_responses = await session.run_role("reasoning", [request])
    synth_responses    = await session.run_role("synthesizer", [synthesis_request])
```

Semantics:

- **Strict declaration.** Calling a role not declared at open raises
  `ValueError`. Aliases canonicalize (declaring `"reasoning"` admits
  `"math"` and `"code"`).
- **Unknown role at open** raises `KeyError` *before* any model is loaded.
- **Unload-on-error.** If the session exits via exception, the currently-
  resident role is torn down so the next session starts cold.
- **Batch-first.** `run_role` takes a list of `PromptRequest`s so multiple
  prompts amortize one load; pass a single-element list for one-shot use.

Use this whenever multiple specialist models cannot co-reside on the
available GPU (e.g., single-A40 deployments). For API-backed or multi-GPU
setups where concurrency is cheap, prefer `get_client(...)` directly —
`FanoutSession` serializes unconditionally.

### `OrchestratedModelClient`

Returned by the named client attributes and by `get_client(...)`.

Methods:

- `await get_response(...)`
- `get_response_sync(...)`

You can either pass a full `PromptRequest` or use the convenience arguments:

- `system_prompt`
- `user_prompt`
- `context`
- `conversation_history`
- `messages`
- `metadata`
- `provider_params`
- `temperature`
- `max_tokens`
- `stop_sequences`

### `OrchestratorRequest`

Wrapper type for passing a prompt plus an optional embedded target:

```python
from llm_gateway import PromptRequest
from model_orchestration import OrchestratorRequest

request = OrchestratorRequest(
    prompt=PromptRequest(user_prompt="Check this claim."),
    target="fever",
)
```

### `OrchestratorResponse`

Structured result returned by orchestrator calls.

Key fields:

- `target`: canonical resolved role
- `resolved_alias`: alias actually used by the caller
- `model`: resolved model id
- `provider`: resolved provider name
- `provider_mode`: `local` or `http` unless explicitly overridden
- `request`: final merged `PromptRequest`
- `prompt_response`: original `llm_gateway.PromptResponse`
- `call_record`: structured event payload
- `recorder_output_path`: JSONL path when a file recorder is active

Convenience properties:

- `text`
- `usage`
- `latency_ms`

### `OrchestratorCallRecord`

Structured metadata record for logging and metrics.

Includes:

- event id
- timestamps
- target / alias / model / provider / provider mode
- normalized request payload
- request metadata
- response text
- usage
- request id
- finish reason
- latency
- response metadata
- raw response
- error type / message for failed calls

## Recording

By default, `ModelOrchestrator` uses `NoOpRecorder`.

### JSONL recording

```python
from pathlib import Path
from model_orchestration import (
    JSONLRecordingConfig,
    ModelOrchestrator,
    build_default_orchestrator_config,
)

config = build_default_orchestrator_config(
    recording=JSONLRecordingConfig(
        output_path=Path("results/orchestration_calls.jsonl")
    )
)

async with ModelOrchestrator(config) as orchestrator:
    await orchestrator.general_client.get_response(user_prompt="Hello")
```

Each completed success or error is appended as one JSON line.

### In-memory recording

Useful for tests:

```python
from model_orchestration import InMemoryRecorder, ModelOrchestrator

recorder = InMemoryRecorder()
orchestrator = ModelOrchestrator(config, recorder=recorder)
```

## Runtime Behavior

### HTTP-backed providers

If the configured provider is not `Provider.LOCAL`, the orchestrator passes the
configured `ProviderConfig` directly to `llm_gateway.create_client(...)`.

### Local provider path

If the configured provider is `Provider.LOCAL`, the orchestrator resolves the
provider config through `resolve_provider_config(...)`, which delegates to the
existing `llm_gateway` local-runtime path.

That means:

- the orchestrator does not implement its own vLLM lifecycle logic
- local runtime behavior stays aligned with the rest of the repo
- tests can replace the resolver with a fake implementation
- one canonical role maps to one local runtime/client
- in the default local helper, those runtimes use separate deterministic ports

### Lazy client creation

The orchestrator creates clients lazily on first use and keeps one managed
client per canonical role until `close()` or context-manager exit.

## Constraints and Notes

- This package is orchestration-only. It does not aggregate model outputs.
- Only configured aliases are valid targets; unknown aliases raise `KeyError`.
- Sync wrappers use `asyncio.run(...)` and fail inside an already-running event loop.
- Request defaults are merged into the outgoing `PromptRequest`; explicit request values win.
- If you want different provider settings per role, define separate `ModelSpec`
  entries with distinct `ProviderConfig` values.
