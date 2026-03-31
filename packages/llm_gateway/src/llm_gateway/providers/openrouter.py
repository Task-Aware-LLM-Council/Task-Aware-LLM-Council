from __future__ import annotations

from llm_gateway.models import ProviderConfig, RetryPolicy, Provider
from llm_gateway.providers.openai_compatible import OpenAICompatibleClient


OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient(OpenAICompatibleClient):
    """Thin OpenRouter wrapper over the OpenAI-compatible client."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        retry_policy: RetryPolicy | None = None,
        http_client=None,
    ) -> None:
        headers = dict(config.headers)
        normalized_config = ProviderConfig(
            provider=Provider.OPENROUTER,
            api_base=config.api_base or OPENROUTER_API_BASE,
            api_key_env=config.api_key_env or "OPENROUTER_API_KEY",
            default_model=config.default_model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            headers=headers,
            default_params=dict(config.default_params),
        )
        super().__init__(
            normalized_config,
            retry_policy=retry_policy,
            http_client=http_client,
        )
