from __future__ import annotations

import logging
from typing import Optional

from llm_gateway import PromptRequest, ProviderConfig, Provider, create_client
from inference.schemas import ModelResponse

logger = logging.getLogger(__name__)


class LLMGateway:
    """Thin async wrapper around llm_gateway for council policy use."""

    def __init__(self, provider_config: Optional[ProviderConfig] = None) -> None:
        self._config = provider_config

    def configure(self, provider_config: ProviderConfig) -> None:
        self._config = provider_config

    async def call(
        self,
        model_id: str,
        prompt: str,
        query_id: str = "",
        role: str = "",
    ) -> ModelResponse:
        config = self._config or ProviderConfig(
            provider=Provider.LOCAL,
            default_model=model_id,
        )
        request = PromptRequest(model=model_id, user_prompt=prompt)
        async with create_client(config) as client:
            response = await client.generate(request)

        return ModelResponse(
            model_id=model_id,
            raw_text=response.text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=response.latency_ms,
        )


gateway = LLMGateway()
