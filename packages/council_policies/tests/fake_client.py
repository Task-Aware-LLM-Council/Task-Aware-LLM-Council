"""
Shared fake LLM client for council_policies tests.

FakeClient takes a fixed `response_text` and returns it for every generate() call.
It also records all requests made so tests can assert on prompts sent.
"""

from __future__ import annotations

from llm_gateway.models import Message, PromptResponse, Usage


class FakeClient:
    """Deterministic fake that returns a preset response."""

    def __init__(self, response_text: str, model_id: str = "fake-model") -> None:
        self.response_text = response_text
        self.model_id = model_id
        self.requests: list = []
        self._closed = False

    async def generate(self, request) -> PromptResponse:
        self.requests.append(request)
        return PromptResponse(
            model=self.model_id,
            text=self.response_text,
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            latency_ms=50.0,
            finish_reason="stop",
        )

    async def close(self) -> None:
        self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()


class FailingClient:
    """Always raises on generate() — used to test error handling."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def generate(self, request):
        raise self.error

    async def close(self) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()


class SequentialClient:
    """Returns responses in order — useful when the same model is called multiple times."""

    def __init__(self, responses: list[str], model_id: str = "fake-model") -> None:
        self._responses = list(responses)
        self._index = 0
        self.model_id = model_id
        self.requests: list = []

    async def generate(self, request) -> PromptResponse:
        self.requests.append(request)
        text = self._responses[self._index % len(self._responses)]
        self._index += 1
        return PromptResponse(
            model=self.model_id,
            text=text,
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            latency_ms=50.0,
            finish_reason="stop",
        )

    async def close(self) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()
