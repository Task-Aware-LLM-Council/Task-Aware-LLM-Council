from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import perf_counter
from typing import Any, Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

from llm_gateway.models import PromptRequest, PromptResponse, ProviderConfig, RetryPolicy


class LLMClientError(Exception):
    """Base exception for provider client failures."""


class LLMRequestError(LLMClientError):
    """Raised when a request is invalid or rejected by the provider."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class LLMRateLimitError(LLMRequestError):
    """Raised when the provider rate-limits a request."""


class LLMTransportError(LLMClientError):
    """Raised when network or transport issues prevent a request."""


class LLMResponseError(LLMClientError):
    """Raised when the provider returns an unusable response payload."""


@dataclass(slots=True, frozen=True)
class HTTPErrorPolicy:
    """Provider-specific classification for non-standard HTTP status semantics."""

    rate_limit_status_codes: tuple[int, ...] = (429,)
    retryable_status_codes: tuple[int, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class ClientInfo:
    provider: str
    default_model: str | None = None
    timeout_seconds: float = 900.0
    max_retries: int = 3
    api_base: str | None = None


class BaseLLMClient(ABC):
    """
    Provider-agnostic base interface for LLM inference clients.

    Concrete clients are responsible for:
    - validating and mapping a request into provider-specific payloads
    - performing the API call
    - normalizing the provider response into `PromptResponse`
    """

    def __init__(
        self,
        info: ClientInfo,
        *,
        config: ProviderConfig | None = None,
        retry_policy: RetryPolicy | None = None,
        error_policy: HTTPErrorPolicy | None = None,
    ) -> None:
        self.info = info
        self.config = config
        self.retry_policy = retry_policy or RetryPolicy(
            max_retries=info.max_retries)
        self.error_policy = error_policy or HTTPErrorPolicy()
        self._closed = False

    @property
    def provider(self) -> str:
        return self.info.provider

    @property
    def default_model(self) -> str | None:
        return self.info.default_model

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> BaseLLMClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Release any client-owned resources."""
        self._closed = True

    def ensure_open(self) -> None:
        if self._closed:
            raise LLMClientError(f"{self.provider} client is closed")

    def resolve_model(self, request: PromptRequest) -> str:
        model = request.model or self.default_model
        if not model:
            raise LLMRequestError(
                f"{self.provider} request is missing a model")
        return model

    def validate_request(self, request: PromptRequest) -> None:
        if not request.resolved_messages():
            raise LLMRequestError(
                f"{self.provider} request must include prompt content")
        self.resolve_model(request)

    def require_api_base(self) -> str:
        api_base = self.info.api_base or (
            self.config.api_base if self.config else None)
        if not api_base:
            raise LLMRequestError(
                f"{self.provider} client is missing api_base")
        return api_base

    def _get_max_attempts(self) -> int:
        return max(self.retry_policy.max_retries, 0) + 1

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        if not self.retry_policy.respect_retry_after:
            return None

        header = response.headers.get("Retry-After")
        if not header:
            return None

        try:
            return max(float(header), 0.0)
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(header)
        except (TypeError, ValueError, IndexError):
            return None

        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)

        return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)

    def _compute_backoff_seconds(
        self,
        attempt_number: int,
        *,
        retry_after_seconds: float | None = None,
    ) -> float:
        if retry_after_seconds is not None:
            base_delay = retry_after_seconds
        else:
            base_delay = min(
                self.retry_policy.initial_backoff_seconds
                * (self.retry_policy.backoff_multiplier **
                   max(attempt_number - 1, 0)),
                self.retry_policy.max_backoff_seconds,
            )

        jitter_window = max(base_delay * self.retry_policy.jitter_ratio, 0.0)
        if jitter_window == 0:
            return base_delay
        return max(base_delay + random.uniform(-jitter_window, jitter_window), 0.0)

    def _make_status_error(self, response: httpx.Response) -> LLMRequestError:
        retry_after_seconds = self._parse_retry_after(response)
        status_code = response.status_code
        message = (
            f"{self.provider} request failed with status {status_code}: {response.text}"
        )
        if status_code in self.error_policy.rate_limit_status_codes:
            return LLMRateLimitError(
                f"{self.provider} rate limited the request",
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
            )
        return LLMRequestError(
            message,
            status_code=status_code,
            retry_after_seconds=retry_after_seconds,
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        if isinstance(error, LLMTransportError): 
            return True

        if isinstance(error, LLMRateLimitError):
            return True

        if isinstance(error, LLMRequestError) and error.status_code is not None:
            return error.status_code in (
                self.retry_policy.retryable_status_codes
                + self.error_policy.retryable_status_codes
            )
                
        return False

    def _with_attempt_metadata(
        self,
        response: PromptResponse,
        *,
        attempt_count: int,
        status_code: int | None = None,
        retry_after_used: float | None = None,
    ) -> PromptResponse:
        metadata = dict(response.metadata)
        metadata["attempt_count"] = attempt_count
        if status_code is not None:
            metadata["status_code"] = status_code
        if retry_after_used is not None:
            metadata["retry_after_used"] = retry_after_used
        return replace(response, metadata=metadata)

    async def _send_with_retries(
        self,
        *,
        send: Callable[[], Awaitable[httpx.Response]],
        parse: Callable[..., PromptResponse],
    ) -> PromptResponse:
        last_error: Exception | None = None
        last_delay_seconds: float | None = None
        max_attempts = self._get_max_attempts()

        for attempt_number in range(1, max_attempts + 1):
            retry_after_seconds: float | None = None
            started = perf_counter()

            try:
                response = await send()
            except httpx.HTTPError as exc:
                error = LLMTransportError(
                    f"{self.provider} transport failure: {exc}")
                last_error = error
            else:
                if response.status_code >= 400:
                    error = self._make_status_error(response)
                    last_error = error
                    retry_after_seconds = getattr(
                        error, "retry_after_seconds", None)
                else:
                    parsed = parse(response, latency_ms=(
                        perf_counter() - started) * 1000)
                    return self._with_attempt_metadata(
                        parsed,
                        attempt_count=attempt_number,
                        status_code=response.status_code,
                        retry_after_used=last_delay_seconds,
                    )
            if last_error is None:
                break
            
            if attempt_number >= max_attempts or not self._is_retryable_error(last_error):
                raise last_error

            delay_seconds = self._compute_backoff_seconds(
                attempt_number,
                retry_after_seconds=retry_after_seconds,
            )
            logger.warning("Request failed, attempt_number:%s, try after:%s, last_error:%s", attempt_number, delay_seconds, last_error)
            last_delay_seconds = delay_seconds
            await asyncio.sleep(delay_seconds)

        logger.error("Request failed completely, last_error:%s", last_error)
        if last_error is None:
            raise LLMClientError(
                f"{self.provider} request failed without an explicit error")

        raise last_error

    @abstractmethod
    async def generate(self, request: PromptRequest) -> PromptResponse:
        """Run one inference request and return a normalized response."""

    async def generate_many(
        self,
        requests: list[PromptRequest],
        *,
        concurrency: int | None = None,
    ) -> list[PromptResponse]:
        """
        Run multiple requests with optional concurrency control.

        This default implementation keeps batching logic in one place so
        benchmark and council pipelines can share it across providers.
        """

        self.ensure_open()
        if not requests:
            return []

        if concurrency is None or concurrency <= 0:
            return list(await asyncio.gather(*(self.generate(request) for request in requests)))

        semaphore = asyncio.Semaphore(concurrency)

        async def _run(request: PromptRequest) -> PromptResponse:
            async with semaphore:
                return await self.generate(request)

        return list(await asyncio.gather(*(_run(request) for request in requests)))

    async def healthcheck(self) -> bool:
        """
        Lightweight readiness hook for concrete clients.

        Providers can override this with a real API probe; the default just
        confirms the client is still open.
        """

        self.ensure_open()
        return True
