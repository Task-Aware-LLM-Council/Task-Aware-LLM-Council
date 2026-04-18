from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any, Callable

from llm_gateway import BaseLLMClient, PromptRequest

from model_orchestration.client import OrchestratedModelClient
from model_orchestration.fanout import FanoutSession
from model_orchestration.models import (
    ModelSpec,
    OrchestratorCallRecord,
    OrchestratorConfig,
    OrchestratorRequest,
    OrchestratorResponse,
)
from model_orchestration.recorders import BaseRecorder, JSONLRecorder, NoOpRecorder
from model_orchestration.runtime import build_client, resolve_provider_config


class _ManagedRoleClient:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        provider_mode: str,
        client_builder: Callable = build_client,
        provider_config_resolver: Callable = resolve_provider_config,
    ) -> None:
        self.spec = spec
        self.provider_mode = provider_mode
        self._client_builder = client_builder
        self._provider_config_resolver = provider_config_resolver
        self._handle_context: AbstractAsyncContextManager | None = None
        self._client: BaseLLMClient | None = None
        self._resolved_provider = spec.provider_config
        self._open_lock = asyncio.Lock()

    async def generate(self, request: PromptRequest):
        if self._client is None:
            await self._ensure_open()
        return await self._client.generate(request)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._handle_context is not None:
            await self._handle_context.__aexit__(None, None, None)
            self._handle_context = None
        self._resolved_provider = self.spec.provider_config

    async def _ensure_open(self) -> None:
        if self._client is not None:
            return
        async with self._open_lock:
            if self._client is not None:
                return
            handle_context = self._provider_config_resolver(
                self.spec.provider_config,
                model=self.spec.model,
            )
            handle = await handle_context.__aenter__()
            try:
                self._resolved_provider = handle.provider_config
                self._client = self._client_builder(handle.provider_config)
            except Exception:
                await handle_context.__aexit__(None, None, None)
                raise
            self._handle_context = handle_context


class ModelOrchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        recorder: BaseRecorder | None = None,
        client_builder: Callable = build_client,
        provider_config_resolver: Callable = resolve_provider_config,
    ) -> None:
        self.config = config
        self.recorder = recorder or self._build_recorder(config)
        self._role_specs: dict[str, ModelSpec] = {}
        self._alias_to_role: dict[str, str] = {}
        self._managed_clients: dict[str, _ManagedRoleClient] = {}
        self._client_builder = client_builder
        self._provider_config_resolver = provider_config_resolver

        for spec in config.models:
            self._register_model(spec)

        self.qa_client = OrchestratedModelClient(self, "qa")
        self.reasoning_client = OrchestratedModelClient(self, "reasoning")
        self.math_client = OrchestratedModelClient(self, "math")
        self.code_client = OrchestratedModelClient(self, "code")
        self.general_client = OrchestratedModelClient(self, "general")
        self.fever_client = OrchestratedModelClient(self, "fever")

    async def __aenter__(self) -> ModelOrchestrator:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def get_client(self, alias: str) -> OrchestratedModelClient:
        self._require_alias(alias)
        return OrchestratedModelClient(self, alias)

    def fanout(self, roles: list[str] | tuple[str, ...]) -> FanoutSession:
        """Open a scoped fan-out session over `roles`.

        Use with `async with`:

            async with orchestrator.fanout(["qa", "reasoning", "general"]) as session:
                qa_out = await session.run_role("qa", [req])
                ...

        Roles are canonicalized via the alias table; passing an unknown
        alias raises KeyError at open time (before any model is loaded).
        """
        canonical = frozenset(self._canonical_role(role) for role in roles)
        return FanoutSession(self, canonical)

    async def run(
        self,
        request: PromptRequest | OrchestratorRequest,
        *,
        target: str | None = None,
    ) -> OrchestratorResponse:
        normalized_request, resolved_alias = self._normalize_run_request(request, target=target)
        role = self._canonical_role(resolved_alias)
        spec = self._role_specs[role]
        managed_client = self._managed_client_for(role)
        prepared_request = _merge_request_defaults(normalized_request, spec)
        request_payload = _request_to_payload(prepared_request)
        event_id = self.recorder.record_request(
            target=role,
            model=spec.model,
            request=request_payload,
        )
        started_at = _utc_now()

        try:
            response = await managed_client.generate(prepared_request)
        except Exception as exc:
            record = OrchestratorCallRecord(
                event_id=event_id,
                started_at=started_at,
                completed_at=_utc_now(),
                target=role,
                resolved_alias=resolved_alias,
                model=spec.model,
                provider=_provider_name(managed_client._resolved_provider.provider),
                provider_mode=_provider_mode(self.config, managed_client._resolved_provider.provider),
                request=request_payload,
                request_metadata=dict(prepared_request.metadata),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            self.recorder.record_error(record)
            raise

        completed_at = _utc_now()
        record = OrchestratorCallRecord(
            event_id=event_id,
            started_at=started_at,
            completed_at=completed_at,
            target=role,
            resolved_alias=resolved_alias,
            model=response.model,
            provider=_provider_name(response.provider or managed_client._resolved_provider.provider),
            provider_mode=_provider_mode(self.config, managed_client._resolved_provider.provider),
            request=request_payload,
            request_metadata=dict(prepared_request.metadata),
            response_text=response.text,
            usage=asdict(response.usage),
            request_id=response.request_id,
            finish_reason=response.finish_reason,
            latency_ms=response.latency_ms,
            response_metadata=dict(response.metadata),
            raw_response=response.raw_response,
        )
        self.recorder.record_response(record)
        return OrchestratorResponse(
            target=role,
            resolved_alias=resolved_alias,
            model=response.model,
            provider=_provider_name(response.provider or managed_client._resolved_provider.provider),
            provider_mode=_provider_mode(self.config, managed_client._resolved_provider.provider),
            request=prepared_request,
            prompt_response=response,
            started_at=started_at,
            completed_at=completed_at,
            call_record=record,
            recorder_output_path=getattr(self.recorder, "output_path", None),
        )

    def run_sync(
        self,
        request: PromptRequest | OrchestratorRequest | None = None,
        *,
        target: str | None = None,
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
        normalized_request = self.build_prompt_request(
            request=request if isinstance(request, PromptRequest) else None,
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
        if isinstance(request, OrchestratorRequest):
            normalized_request = request
        return _run_sync(self.run(normalized_request or request, target=target))

    async def close(self) -> None:
        for managed_client in self._managed_clients.values():
            await managed_client.close()

    def build_prompt_request(
        self,
        *,
        request: PromptRequest | None = None,
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
    ) -> PromptRequest:
        if request is not None:
            return request
        return PromptRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            conversation_history=tuple(conversation_history),
            messages=tuple(messages),
            metadata=dict(metadata or {}),
            provider_params=dict(provider_params or {}),
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=tuple(stop_sequences),
        )

    def _register_model(self, spec: ModelSpec) -> None:
        role = spec.role.strip().lower()
        if role in self._role_specs:
            raise ValueError(f"Duplicate role configured: {role}")
        self._role_specs[role] = replace(spec, role=role)
        aliases = {role, *(alias.strip().lower() for alias in spec.aliases)}
        for alias in aliases:
            if alias in self._alias_to_role:
                raise ValueError(f"Duplicate alias configured: {alias}")
            self._alias_to_role[alias] = role

    def _managed_client_for(self, role: str) -> _ManagedRoleClient:
        managed = self._managed_clients.get(role)
        if managed is None:
            managed = _ManagedRoleClient(
                self._role_specs[role],
                provider_mode=_provider_mode(self.config, self._role_specs[role].provider_config.provider),
                client_builder=self._client_builder,
                provider_config_resolver=self._provider_config_resolver,
            )
            self._managed_clients[role] = managed
        return managed

    def _normalize_run_request(
        self,
        request: PromptRequest | OrchestratorRequest,
        *,
        target: str | None,
    ) -> tuple[PromptRequest, str]:
        if isinstance(request, OrchestratorRequest):
            resolved_alias = target or request.target or self.config.default_role
            return request.prompt, resolved_alias
        return request, target or self.config.default_role

    def _canonical_role(self, alias: str) -> str:
        return self._alias_to_role[self._require_alias(alias)]

    def _require_alias(self, alias: str) -> str:
        normalized = alias.strip().lower()
        if normalized not in self._alias_to_role:
            raise KeyError(f"Unknown orchestrator target: {alias}")
        return normalized

    @staticmethod
    def _build_recorder(config: OrchestratorConfig) -> BaseRecorder:
        if config.recording is None:
            return NoOpRecorder()
        return JSONLRecorder(config.recording.output_path)


def _merge_request_defaults(request: PromptRequest, spec: ModelSpec) -> PromptRequest:
    defaults = dict(spec.request_defaults)
    metadata = dict(defaults.pop("metadata", {}))
    metadata.update(request.metadata)
    provider_params = dict(defaults.pop("provider_params", {}))
    provider_params.update(request.provider_params)

    merged = replace(
        request,
        model=request.model or spec.model,
        metadata=metadata,
        provider_params=provider_params,
    )

    for field_name in ("temperature", "max_tokens", "system_prompt", "user_prompt", "context"):
        default_value = defaults.get(field_name)
        if getattr(merged, field_name) is None and default_value is not None:
            merged = replace(merged, **{field_name: default_value})

    if not merged.stop_sequences and defaults.get("stop_sequences"):
        merged = replace(merged, stop_sequences=tuple(defaults["stop_sequences"]))

    return merged


def _request_to_payload(request: PromptRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "messages": [asdict(message) for message in request.resolved_messages()],
        "system_prompt": request.system_prompt,
        "user_prompt": request.user_prompt,
        "context": request.context,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stop_sequences": list(request.stop_sequences),
        "metadata": dict(request.metadata),
        "provider_params": dict(request.provider_params),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Sync orchestration APIs cannot run inside an active event loop")


def _provider_mode(config: OrchestratorConfig, provider) -> str:
    if config.mode_label:
        return config.mode_label
    normalized = _provider_name(provider)
    if normalized == "local":
        return "local"
    return "http"


def _provider_name(provider) -> str:
    return getattr(provider, "value", str(provider))
