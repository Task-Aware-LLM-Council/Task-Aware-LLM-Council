from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


class Provider(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai-compatible"
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"


@dataclass(slots=True, frozen=True)
class Message:
    role: Role
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PromptRequest:
    model: str | None = None
    messages: tuple[Message, ...] = ()
    system_prompt: str | None = None
    user_prompt: str | None = None
    context: str | None = None
    conversation_history: tuple[Message, ...] = ()
    temperature: float | None = None
    max_tokens: int | None = None
    stop_sequences: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_params: dict[str, Any] = field(default_factory=dict)

    def resolved_messages(self) -> tuple[Message, ...]:
        """
        Return a normalized message sequence for chat-style providers.

        If explicit `messages` are supplied, they are treated as the source of
        truth. Otherwise the request is assembled from prompt convenience fields.
        """

        if self.messages:
            return self.messages

        normalized_messages: list[Message] = []

        if self.system_prompt:
            normalized_messages.append(Message(role="system", content=self.system_prompt))

        normalized_user_parts: list[str] = []
        if self.context:
            normalized_user_parts.append(self.context)
        if self.user_prompt:
            normalized_user_parts.append(self.user_prompt)

        normalized_messages.extend(self.conversation_history)

        if normalized_user_parts:
            normalized_messages.append(
                Message(role="user", content="\n\n".join(normalized_user_parts))
            )

        return tuple(normalized_messages)


@dataclass(slots=True, frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    currency: str | None = None


@dataclass(slots=True, frozen=True)
class ResponseChoice:
    index: int
    message: Message
    finish_reason: str | None = None
    score: float | None = None


@dataclass(slots=True, frozen=True)
class PromptResponse:
    model: str
    text: str
    choices: tuple[ResponseChoice, ...] = ()
    usage: Usage = field(default_factory=Usage)
    latency_ms: float | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    provider: str | None = None
    raw_response: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ProviderConfig:
    provider: Provider | str
    api_base: str | None = None
    api_key_env: str | None = None
    default_model: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    headers: dict[str, str] = field(default_factory=dict)
    default_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RetryPolicy:
    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.1
    respect_retry_after: bool = True
    retryable_status_codes: tuple[int, ...] = (408, 409, 429, 500, 502, 503, 504)
