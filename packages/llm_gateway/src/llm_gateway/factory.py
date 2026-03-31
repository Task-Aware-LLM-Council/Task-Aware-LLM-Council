from __future__ import annotations

from llm_gateway.base import BaseLLMClient, HTTPErrorPolicy, LLMClientError
from llm_gateway.models import Provider, ProviderConfig, RetryPolicy
from llm_gateway.providers.openai import OpenAIClient
from llm_gateway.providers.openai_compatible import OpenAICompatibleClient
from llm_gateway.providers.openrouter import OpenRouterClient

HUGGINGFACE_ROUTER_API_BASE = "https://router.huggingface.co/v1/chat/completions"
HUGGINGFACE_ERROR_POLICY = HTTPErrorPolicy(rate_limit_status_codes=(402, 429))


def normalize_provider_name(provider: Provider | str) -> str:
    if isinstance(provider, Provider):
        return provider.value
    return provider.strip().lower()


def create_client(
    config: ProviderConfig,
    *,
    retry_policy: RetryPolicy | None = None,
) -> BaseLLMClient:
    provider = normalize_provider_name(config.provider)

    if provider == Provider.OPENROUTER.value:
        return OpenRouterClient(config, retry_policy=retry_policy)

    if provider == Provider.OPENAI.value:
        return OpenAIClient(config, retry_policy=retry_policy)

    if provider == Provider.OPENAI_COMPATIBLE.value:
        return OpenAICompatibleClient(config, retry_policy=retry_policy)

    if provider in {Provider.HUGGINGFACE.value, "hf"}:
        normalized_config = ProviderConfig(
            provider=Provider.HUGGINGFACE,
            api_base=config.api_base or HUGGINGFACE_ROUTER_API_BASE,
            api_key_env=config.api_key_env or "HUGGINGFACE_API_KEY",
            default_model=config.default_model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            headers=dict(config.headers),
            default_params=dict(config.default_params),
        )
        return OpenAICompatibleClient(
            normalized_config,
            retry_policy=retry_policy,
            error_policy=HUGGINGFACE_ERROR_POLICY,
        )

    raise LLMClientError(f"Unsupported provider: {config.provider}")
