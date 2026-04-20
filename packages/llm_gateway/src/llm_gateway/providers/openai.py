from __future__ import annotations

from typing import Any

from llm_gateway.models import ProviderConfig, RetryPolicy, Provider
from llm_gateway.providers.openai_compatible import OpenAICompatibleClient


OPENAI_API_BASE = "https://api.openai.com/v1/chat/completions"


class OpenAIClient(OpenAICompatibleClient):
    """Thin OpenAI wrapper over the OpenAI-compatible client."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        retry_policy: RetryPolicy | None = None,
        http_client: Any = None,
    ) -> None:
        normalized_config = ProviderConfig(
            provider=Provider.OPENAI,
            api_base=config.api_base or OPENAI_API_BASE,
            api_key_env=config.api_key_env or "OPENAI_API_KEY",
            default_model=config.default_model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            headers=dict(config.headers),
            default_params=dict(config.default_params),
        )
        super().__init__(
            normalized_config,
            retry_policy=retry_policy,
            http_client=http_client,
        )
