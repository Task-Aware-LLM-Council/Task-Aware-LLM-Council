from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol, TYPE_CHECKING

from llm_gateway import Message, PromptRequest, PromptResponse
from model_orchestration import ModelOrchestrator, OrchestratorConfig, OrchestratorResponse

if TYPE_CHECKING:
    from council_policies.router import DispatchRun


def request_fingerprint(request: PromptRequest) -> str:
    payload = {
        "model": request.model,
        "messages": [_message_payload(message) for message in request.messages],
        "system_prompt": request.system_prompt,
        "user_prompt": request.user_prompt,
        "context": request.context,
        "conversation_history": [
            _message_payload(message) for message in request.conversation_history
        ],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stop_sequences": list(request.stop_sequences),
        "metadata": request.metadata,
        "provider_params": request.provider_params,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def merge_prompt_response_metadata(
    response: PromptResponse,
    metadata: dict[str, Any],
) -> PromptResponse:
    merged = dict(response.metadata)
    merged.update(metadata)
    return replace(response, metadata=merged)


def orchestrator_to_prompt_response(
    response: OrchestratorResponse,
    metadata: dict[str, Any] | None = None,
) -> PromptResponse:
    prompt_response = response.prompt_response
    if not metadata:
        return prompt_response
    return merge_prompt_response_metadata(prompt_response, metadata)


@dataclass(slots=True, frozen=True)
class SpecialistRequest:
    cache_key: str
    role: str
    request: PromptRequest


@dataclass(slots=True)
class PolicyExecutionState:
    policy_id: str
    request: PromptRequest
    specialist_requests: list[SpecialistRequest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    final_response: PromptResponse | None = None
    winner_response: OrchestratorResponse | None = None


@dataclass(slots=True, frozen=True)
class PolicyResult:
    policy_id: str
    request: PromptRequest
    response: PromptResponse
    metadata: dict[str, Any] = field(default_factory=dict)


class SpecialistCache:
    def __init__(self) -> None:
        self._responses: dict[str, OrchestratorResponse] = {}

    def has(self, cache_key: str) -> bool:
        return cache_key in self._responses

    def get(self, cache_key: str) -> OrchestratorResponse:
        return self._responses[cache_key]

    def put(self, cache_key: str, response: OrchestratorResponse) -> None:
        self._responses[cache_key] = response

    def keys(self) -> tuple[str, ...]:
        return tuple(self._responses)


class CouncilPolicyAdapter(Protocol):
    policy_id: str

    async def plan(
        self,
        request: PromptRequest,
        runtime: PolicyRuntime,
    ) -> PolicyExecutionState: ...

    async def complete_specialist_phase(
        self,
        state: PolicyExecutionState,
        cache: SpecialistCache,
        runtime: PolicyRuntime,
    ) -> PolicyExecutionState: ...

    async def finalize(
        self,
        state: PolicyExecutionState,
        cache: SpecialistCache,
        runtime: PolicyRuntime,
    ) -> PolicyResult: ...


@dataclass(slots=True, frozen=True)
class PolicyBenchmarkResult:
    results: tuple[PolicyResult, ...]
    specialist_cache_keys: tuple[str, ...]


class PolicyRuntime:
    def __init__(
        self,
        *,
        specialist_config: OrchestratorConfig,
        synthesizer_config: OrchestratorConfig,
        specialist_load_parallel: int = 1,
        synthesizer_load_parallel: int = 1,
        orchestrator_factory=ModelOrchestrator,
    ) -> None:
        self.specialist_config = specialist_config
        self.synthesizer_config = synthesizer_config
        self.specialist_load_parallel = specialist_load_parallel
        self.synthesizer_load_parallel = synthesizer_load_parallel
        self._orchestrator_factory = orchestrator_factory
        self._specialist_orchestrator: Any | None = None
        self._synthesizer_orchestrator: Any | None = None

    def has_specialist_role(self, role: str) -> bool:
        return role in self._known_roles(self.specialist_config)

    def has_synthesizer_role(self, role: str) -> bool:
        return role in self._known_roles(self.synthesizer_config)

    async def open_specialists(self) -> Any:
        if self._specialist_orchestrator is None:
            orchestrator = self._orchestrator_factory(self.specialist_config)
            self._specialist_orchestrator = await orchestrator.__aenter__()
            await self._specialist_orchestrator.load_all(
                max_parallel=self.specialist_load_parallel
            )
        return self._specialist_orchestrator

    async def close_specialists(self) -> None:
        if self._specialist_orchestrator is not None:
            await self._specialist_orchestrator.__aexit__(None, None, None)
            self._specialist_orchestrator = None

    async def get_synthesizer_orchestrator(self) -> Any:
        if self._synthesizer_orchestrator is None:
            orchestrator = self._orchestrator_factory(self.synthesizer_config)
            self._synthesizer_orchestrator = await orchestrator.__aenter__()
            await self._synthesizer_orchestrator.load_all(
                max_parallel=self.synthesizer_load_parallel
            )
        return self._synthesizer_orchestrator

    async def close_synthesizer(self) -> None:
        if self._synthesizer_orchestrator is not None:
            await self._synthesizer_orchestrator.__aexit__(None, None, None)
            self._synthesizer_orchestrator = None

    @property
    def specialist_orchestrator(self) -> Any:
        if self._specialist_orchestrator is None:
            raise RuntimeError("specialist orchestrator is not open")
        return self._specialist_orchestrator

    async def execute_specialist_requests(
        self,
        specialist_requests: list[SpecialistRequest],
        cache: SpecialistCache,
    ) -> None:
        if not specialist_requests:
            return

        await self.open_specialists()

        unique_requests: list[SpecialistRequest] = []
        seen: set[str] = set()
        for specialist_request in specialist_requests:
            if cache.has(specialist_request.cache_key):
                continue
            if specialist_request.cache_key in seen:
                continue
            seen.add(specialist_request.cache_key)
            unique_requests.append(specialist_request)

        async def _run(specialist_request: SpecialistRequest) -> None:
            response = await self.specialist_orchestrator.get_client(
                specialist_request.role
            ).get_response(specialist_request.request)
            cache.put(specialist_request.cache_key, response)

        await asyncio.gather(*(_run(request) for request in unique_requests))

    async def execute_vote_request(
        self,
        *,
        role: str,
        request: PromptRequest,
    ) -> OrchestratorResponse:
        await self.open_specialists()
        return await self.specialist_orchestrator.get_client(role).get_response(request)

    async def execute_vote_requests(
        self,
        *,
        voter_roles: tuple[str, ...],
        request: PromptRequest,
    ) -> list[OrchestratorResponse]:
        await self.open_specialists()
        return await asyncio.gather(
            *(
                self.specialist_orchestrator.get_client(role).get_response(request)
                for role in voter_roles
            )
        )

    async def close(self) -> None:
        await self.close_specialists()
        await self.close_synthesizer()

    async def __aenter__(self) -> PolicyRuntime:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @staticmethod
    def _known_roles(config: OrchestratorConfig) -> set[str]:
        known: set[str] = set()
        for spec in config.models:
            known.add(str(spec.role))
            known.update(str(alias) for alias in getattr(spec, "aliases", ()))
        return known


class BasePolicyAdapter:
    async def complete_specialist_phase(
        self,
        state: PolicyExecutionState,
        cache: SpecialistCache,
        runtime: PolicyRuntime,
    ) -> PolicyExecutionState:
        del cache, runtime
        return state


class CouncilBenchmarkRunner:
    def __init__(
        self,
        *,
        policies: tuple[CouncilPolicyAdapter, ...],
        runtime: PolicyRuntime,
    ) -> None:
        self.policies = policies
        self.runtime = runtime

    async def run(
        self,
        requests: list[PromptRequest],
    ) -> PolicyBenchmarkResult:
        cache = SpecialistCache()
        states: list[tuple[CouncilPolicyAdapter, PolicyExecutionState]] = []

        for request in requests:
            for policy in self.policies:
                state = await policy.plan(request, self.runtime)
                states.append((policy, state))

        all_specialist_requests = [
            specialist_request
            for _, state in states
            for specialist_request in state.specialist_requests
        ]
        await self.runtime.execute_specialist_requests(all_specialist_requests, cache)

        updated_states: list[tuple[CouncilPolicyAdapter, PolicyExecutionState]] = []
        for policy, state in states:
            updated_state = await policy.complete_specialist_phase(
                state, cache, self.runtime
            )
            updated_states.append((policy, updated_state))

        await self.runtime.close_specialists()

        results: list[PolicyResult] = []
        for policy, state in updated_states:
            results.append(await policy.finalize(state, cache, self.runtime))

        await self.runtime.close_synthesizer()
        return PolicyBenchmarkResult(
            results=tuple(results),
            specialist_cache_keys=cache.keys(),
        )


def build_specialist_cache_key(
    policy_id: str,
    request: PromptRequest,
    role: str,
    *,
    phase: str,
    prompt_override: PromptRequest | None = None,
    extra: str | None = None,
) -> str:
    request_id = str(request.metadata.get("example_id") or request_fingerprint(request))
    effective_request = prompt_override or request
    parts = [policy_id, request_id, phase, role, request_fingerprint(effective_request)]
    if extra:
        parts.append(extra)
    return "::".join(parts)


def state_metadata_with_response(
    state: PolicyExecutionState,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(state.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata.setdefault("policy_id", state.policy_id)
    metadata.setdefault("specialist_cache_keys", [req.cache_key for req in state.specialist_requests])
    return metadata


def make_policy_result(
    state: PolicyExecutionState,
    *,
    response: PromptResponse,
    extra_metadata: dict[str, Any] | None = None,
) -> PolicyResult:
    metadata = state_metadata_with_response(state, extra_metadata=extra_metadata)
    return PolicyResult(
        policy_id=state.policy_id,
        request=state.request,
        response=merge_prompt_response_metadata(response, metadata),
        metadata=metadata,
    )


def build_run_from_cached_response(
    role: str,
    subtask_texts: list[dict[str, Any]],
    response: OrchestratorResponse,
) -> DispatchRun:
    from council_policies.router import DispatchRun, Subtask

    run = DispatchRun(role=role)
    for subtask_payload in subtask_texts:
        run.append(
            Subtask(
                text=str(subtask_payload["text"]),
                order=int(subtask_payload["order"]),
            )
        )
    run.response = response
    return run


def prompt_request_from_messages(
    *,
    messages: tuple[Message, ...] = (),
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    context: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PromptRequest:
    return PromptRequest(
        messages=messages,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context=context,
        metadata=metadata or {},
    )


def serialise_orchestrator_response(response: OrchestratorResponse) -> dict[str, Any]:
    return {
        "target": response.target,
        "resolved_alias": response.resolved_alias,
        "model": response.model,
        "provider": response.provider,
        "provider_mode": response.provider_mode,
        "prompt_response": asdict(response.prompt_response),
        "started_at": response.started_at,
        "completed_at": response.completed_at,
    }


def _message_payload(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "name": message.name,
        "metadata": message.metadata,
    }
