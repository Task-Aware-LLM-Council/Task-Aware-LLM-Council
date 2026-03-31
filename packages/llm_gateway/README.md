# llm_gateway

`llm_gateway` is a small reusable client layer for text-generation style LLM calls. It is intended to be the shared API surface for benchmarking now and council-style workflows later.

## What Exists Today

- Normalized request/response models for prompt execution
- Shared retry, backoff, and rate-limit handling in the base client layer
- Thin provider wrappers for direct OpenAI and OpenRouter
- Generic OpenAI-compatible transport for any OpenAI-style endpoint
- Config-driven client construction through `create_client(...)`
- Unit compatibility tests for the current public surface
- Smoke-test hooks for OpenAI, OpenRouter, and Hugging Face router

The current supported provider paths are:

- `OpenAIClient` for direct OpenAI chat completions
- `OpenRouterClient` for OpenRouter’s OpenAI-style gateway
- `OpenAICompatibleClient` for Hugging Face router and any other OpenAI-style endpoint
- `create_client(...)` for config-driven construction, including `"huggingface"` / `"hf"` mapping to Hugging Face router defaults

## Public API

Import from the package root:

```python
from llm_gateway import OpenRouterClient, PromptRequest, ProviderConfig
```

### Clients

- `OpenAIClient`
- `OpenRouterClient`
- `OpenAICompatibleClient`
- `BaseLLMClient`

### Factory

- `create_client`

### Request / Response Types

- `Message`
- `PromptRequest`
- `PromptResponse`
- `ProviderConfig`
- `RetryPolicy`
- `ResponseChoice`
- `Usage`

### Exceptions

- `LLMClientError`
- `LLMRequestError`
- `LLMRateLimitError`
- `LLMResponseError`
- `LLMTransportError`

## Recommended Usage

- Use `OpenRouterClient` as the default path when you want one client across many hosted models.
- Use `OpenAIClient` only when you want direct OpenAI access.
- Use `OpenAICompatibleClient` when the endpoint already speaks the OpenAI chat-completions protocol.
- Use `create_client(...)` when the caller wants provider selection from config instead of hardcoding a class.

## Examples

```python
from llm_gateway import OpenRouterClient, PromptRequest, ProviderConfig

client = OpenRouterClient(
    ProviderConfig(
        provider=Provider.OPENROUTER,
        default_model="openai/gpt-4o-mini",
    )
)

response = await client.generate(
    PromptRequest(
        user_prompt="Give a one sentence summary of testing.",
        temperature=0.0,
        max_tokens=64,
    )
)

print(response.text)
await client.close()
```

```python
from llm_gateway import OpenAIClient, PromptRequest, ProviderConfig

client = OpenAIClient(
    ProviderConfig(
        provider=Provider.OPENAI,
        default_model="gpt-4o-mini",
    )
)

response = await client.generate(PromptRequest(user_prompt="Say hello."))
print(response.text)
await client.close()
```

```python
from llm_gateway import create_client, PromptRequest, ProviderConfig

client = create_client(
    ProviderConfig(
        provider=Provider.HUGGINGFACE,
        default_model="Qwen/Qwen3-32B",
    )
)

response = await client.generate(
    PromptRequest(
        user_prompt="Say hello.",
    )
)

print(response.text)
await client.close()
```

```python
from llm_gateway import OpenAICompatibleClient, PromptRequest, ProviderConfig

client = OpenAICompatibleClient(
    ProviderConfig(
        provider=Provider.HUGGINGFACE,
        api_base="https://router.huggingface.co/v1/chat/completions",
        api_key_env="HUGGINGFACE_API_KEY",
        default_model="Qwen/Qwen3-32B",
    )
)

response = await client.generate(PromptRequest(user_prompt="Say hello."))
print(response.text)
await client.close()
```

## Environment Variables

- `OPENAI_API_KEY` for `OpenAIClient`
- `OPENROUTER_API_KEY` for `OpenRouterClient`
- `HUGGINGFACE_API_KEY` for Hugging Face router via `OpenAICompatibleClient` or `create_client(...)`

Smoke tests additionally use:

- `OPENAI_SMOKE_MODEL`
- `OPENROUTER_SMOKE_MODEL`
- `HUGGINGFACE_SMOKE_MODEL`

## Running Tests

Install test dependencies:

```bash
uv sync --extra test
```

Run compatibility tests:

```bash
uv run pytest packages/llm_gateway/tests -m unit
```

Run live smoke tests:

```bash
uv run pytest packages/llm_gateway/tests/test_smoke.py -m smoke -rs
```

Smoke tests skip automatically when required environment variables are not set.

## Not Finished Yet

- streaming responses
- tool/function calling support
- richer provider-specific features beyond text generation
- multimodal request/response support
