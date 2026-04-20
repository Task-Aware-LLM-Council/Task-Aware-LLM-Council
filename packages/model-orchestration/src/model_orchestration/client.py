from __future__ import annotations

from typing import Any

from llm_gateway import PromptRequest

from model_orchestration.models import OrchestratorResponse


class OrchestratedModelClient:
    def __init__(self, orchestrator, alias: str) -> None:
        self._orchestrator = orchestrator
        self._alias = alias

    async def get_response(
        self,
        request: PromptRequest | None = None,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        context: str | None = None,
        conversation_history=(),
        messages=(),
        metadata: dict[str, Any] | None = None,
        provider_params: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: tuple[str, ...] = (),
    ) -> OrchestratorResponse:
        normalized_request = self._orchestrator.build_prompt_request(
            request=request,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            conversation_history=conversation_history,
            messages=messages,
            metadata=metadata,
            provider_params=provider_params,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )
        return await self._orchestrator.run(normalized_request, target=self._alias)

    def get_response_sync(
        self,
        request: PromptRequest | None = None,
        **kwargs: Any,
    ) -> OrchestratorResponse:
        return self._orchestrator.run_sync(request, target=self._alias, **kwargs)
