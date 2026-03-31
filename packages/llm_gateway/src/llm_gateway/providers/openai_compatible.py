from __future__ import annotations

import os
from typing import Any

import httpx

from llm_gateway.base import (
    BaseLLMClient,
    ClientInfo,
    HTTPErrorPolicy,
    LLMResponseError,
)
from llm_gateway.models import (
    Message,
    PromptRequest,
    PromptResponse,
    ProviderConfig,
    ResponseChoice,
    RetryPolicy,
    Usage,
)


class OpenAICompatibleClient(BaseLLMClient):
    """
    Generic client for OpenAI-style chat completion APIs.

    This is intentionally generic enough to back OpenAI, OpenRouter, and
    other providers that expose a similar `/chat/completions` interface.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        retry_policy: RetryPolicy | None = None,
        error_policy: HTTPErrorPolicy | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        info = ClientInfo(
            provider=config.provider,
            default_model=config.default_model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            api_base=config.api_base,
        )
        super().__init__(
            info,
            config=config,
            retry_policy=retry_policy,
            error_policy=error_policy,
        )
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=config.timeout_seconds,
            headers=config.headers,
        )

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()
        await super().close()

    def _get_api_key(self) -> str | None:
        if not self.config or not self.config.api_key_env:
            return None
        return os.getenv(self.config.api_key_env)

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers) if self.config else {}
        api_key = self._get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.setdefault("Content-Type", "application/json")
        return headers

    def _build_payload(self, request: PromptRequest) -> dict[str, Any]:
        self.validate_request(request)
        model = self.resolve_model(request)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                self._message_to_payload(message)
                for message in request.resolved_messages()
            ],
        }

        if self.config:
            payload.update(self.config.default_params)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            payload["stop"] = list(request.stop_sequences)
        payload.update(request.provider_params)
        return payload

    @staticmethod
    def _message_to_payload(message: Message) -> dict[str, Any]:
        payload = {
            "role": message.role,
            "content": message.content,
        }
        if message.name:
            payload["name"] = message.name
        return payload

    async def generate(self, request: PromptRequest) -> PromptResponse:
        self.ensure_open()
        payload = self._build_payload(request)
        api_base = self.require_api_base()
        return await self._send_with_retries(
            send=lambda: self._http_client.post(
                api_base,
                json=payload,
                headers=self._build_headers(),
            ),
            parse=self._parse_response,
        )

    def _parse_response(
        self,
        response: httpx.Response,
        *,
        latency_ms: float,
    ) -> PromptResponse:
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMResponseError(f"{self.provider} returned non-JSON response") from exc

        choices_payload = payload.get("choices", [])
        if not choices_payload:
            raise LLMResponseError(f"{self.provider} response did not include choices")

        choices: list[ResponseChoice] = []
        for index, choice_payload in enumerate(choices_payload):
            message_payload = choice_payload.get("message", {})
            message = Message(
                role=message_payload.get("role", "assistant"),
                content=message_payload.get("content", ""),
            )
            choices.append(
                ResponseChoice(
                    index=index,
                    message=message,
                    finish_reason=choice_payload.get("finish_reason"),
                )
            )

        first_choice = choices[0]
        usage_payload = payload.get("usage", {})
        usage = Usage(
            input_tokens=usage_payload.get("prompt_tokens"),
            output_tokens=usage_payload.get("completion_tokens"),
            total_tokens=usage_payload.get("total_tokens"),
        )
        return PromptResponse(
            model=payload.get("model", self.default_model or "unknown"),
            text=first_choice.message.content,
            choices=tuple(choices),
            usage=usage,
            latency_ms=latency_ms,
            request_id=payload.get("id"),
            finish_reason=first_choice.finish_reason,
            provider=self.provider,
            raw_response=payload,
        )
