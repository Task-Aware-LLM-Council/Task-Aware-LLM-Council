from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from llm_gateway import (
    PromptRequest,
    PromptResponse,
    Provider,
    ProviderConfig,
    Usage,
)
from model_orchestration import (
    InMemoryRecorder,
    ModelOrchestrator,
    ModelSpec,
    OrchestratorConfig,
)
from model_orchestration.runtime import ResolvedProviderHandle


class FakeClient:
    def __init__(self, config: ProviderConfig, *, fail: bool = False) -> None:
        self.config = config
        self.fail = fail
        self.requests: list[PromptRequest] = []
        self.closed = False

    async def generate(self, request: PromptRequest) -> PromptResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError(f"boom:{self.config.default_model}")
        return PromptResponse(
            model=request.model or self.config.default_model or "unknown",
            text=f"response:{self.config.default_model}",
            latency_ms=1.0,
            request_id=f"req-{self.config.default_model}",
            finish_reason="stop",
            provider=getattr(self.config.provider, "value", str(self.config.provider)),
            usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            metadata={},
            raw_response={},
        )

    async def close(self) -> None:
        self.closed = True


def _config(provider=Provider.OPENAI_COMPATIBLE) -> OrchestratorConfig:
    return OrchestratorConfig(
        models=(
            ModelSpec(
                role="qa",
                model="qa-model",
                aliases=("qa",),
                provider_config=ProviderConfig(
                    provider=provider,
                    api_base="http://qa.test/v1",
                    default_model="qa-model",
                ),
            ),
            ModelSpec(
                role="reasoning",
                model="reasoning-model",
                aliases=("reasoning", "math", "code"),
                provider_config=ProviderConfig(
                    provider=provider,
                    api_base="http://reasoning.test/v1",
                    default_model="reasoning-model",
                ),
            ),
            ModelSpec(
                role="general",
                model="general-model",
                aliases=("general", "fever"),
                provider_config=ProviderConfig(
                    provider=provider,
                    api_base="http://general.test/v1",
                    default_model="general-model",
                ),
            ),
        ),
        default_role="general",
    )


def _build_orchestrator(*, fail_roles: set[str] | None = None) -> tuple[ModelOrchestrator, list[FakeClient]]:
    fail_roles = fail_roles or set()
    clients: list[FakeClient] = []

    def client_builder(provider_config: ProviderConfig) -> FakeClient:
        client = FakeClient(
            provider_config,
            fail=provider_config.default_model in fail_roles,
        )
        clients.append(client)
        return client

    orchestrator = ModelOrchestrator(
        _config(),
        recorder=InMemoryRecorder(),
        client_builder=client_builder,
    )
    return orchestrator, clients


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fanout_runs_declared_role_and_returns_responses() -> None:
    orchestrator, clients = _build_orchestrator()

    async with orchestrator.fanout(["qa", "reasoning", "general"]) as session:
        responses = await session.run_role(
            "qa",
            [PromptRequest(user_prompt="hello")],
        )

    assert len(responses) == 1
    assert responses[0].target == "qa"
    assert responses[0].text == "response:qa-model"
    assert len(clients) == 1
    # Clean exit unloads the last-used role.
    assert clients[0].closed is True

    await orchestrator.close()


@pytest.mark.asyncio
async def test_fanout_swaps_on_role_change_closing_previous_client() -> None:
    orchestrator, clients = _build_orchestrator()

    async with orchestrator.fanout(["qa", "reasoning"]) as session:
        await session.run_role("qa", [PromptRequest(user_prompt="one")])
        assert len(clients) == 1
        assert clients[0].closed is False  # still resident

        await session.run_role("reasoning", [PromptRequest(user_prompt="two")])
        # Swap fired: qa client closed, reasoning client created.
        assert len(clients) == 2
        assert clients[0].closed is True
        assert clients[1].closed is False

    # Clean exit unloads the currently-resident role.
    assert clients[1].closed is True

    await orchestrator.close()


@pytest.mark.asyncio
async def test_fanout_same_role_twice_does_not_reswap() -> None:
    orchestrator, clients = _build_orchestrator()

    async with orchestrator.fanout(["qa"]) as session:
        await session.run_role("qa", [PromptRequest(user_prompt="a")])
        await session.run_role("qa", [PromptRequest(user_prompt="b")])

    # Same role called twice: one client created, reused, then closed on exit.
    assert len(clients) == 1
    assert len(clients[0].requests) == 2
    assert clients[0].closed is True

    await orchestrator.close()


@pytest.mark.asyncio
async def test_fanout_batches_multiple_requests_within_one_role() -> None:
    orchestrator, clients = _build_orchestrator()

    requests = [PromptRequest(user_prompt=f"q{i}") for i in range(3)]
    async with orchestrator.fanout(["qa"]) as session:
        responses = await session.run_role("qa", requests)

    assert len(responses) == 3
    assert len(clients) == 1
    assert len(clients[0].requests) == 3

    await orchestrator.close()


# --------------------------------------------------------------------------- #
# Strict declaration
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fanout_undeclared_role_raises_before_loading() -> None:
    orchestrator, clients = _build_orchestrator()

    async with orchestrator.fanout(["qa"]) as session:
        with pytest.raises(ValueError, match="was not declared"):
            await session.run_role("reasoning", [PromptRequest(user_prompt="nope")])

    # Nothing should have been loaded for reasoning.
    assert all(c.config.default_model == "qa-model" for c in clients)

    await orchestrator.close()


@pytest.mark.asyncio
async def test_fanout_alias_is_canonicalized_in_declaration_check() -> None:
    # Declaring "reasoning" should admit calls via its aliases ("math", "code").
    orchestrator, clients = _build_orchestrator()

    async with orchestrator.fanout(["reasoning"]) as session:
        await session.run_role("math", [PromptRequest(user_prompt="2+2")])
        await session.run_role("code", [PromptRequest(user_prompt="print hi")])

    # All calls hit the one canonical role -> one client.
    assert len(clients) == 1
    assert clients[0].config.default_model == "reasoning-model"
    assert len(clients[0].requests) == 2

    await orchestrator.close()


@pytest.mark.asyncio
async def test_fanout_unknown_role_at_open_raises_key_error() -> None:
    orchestrator, _ = _build_orchestrator()

    with pytest.raises(KeyError):
        orchestrator.fanout(["nonexistent"])

    await orchestrator.close()


# --------------------------------------------------------------------------- #
# Exception path: unload-on-error
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fanout_unloads_current_role_on_exception() -> None:
    orchestrator, clients = _build_orchestrator()

    with pytest.raises(RuntimeError, match="policy-level failure"):
        async with orchestrator.fanout(["qa"]) as session:
            await session.run_role("qa", [PromptRequest(user_prompt="hi")])
            assert clients[0].closed is False
            raise RuntimeError("policy-level failure")

    # Unload-on-error policy: current role must be closed.
    assert clients[0].closed is True

    await orchestrator.close()


@pytest.mark.asyncio
async def test_fanout_unloads_when_run_role_itself_raises() -> None:
    orchestrator, clients = _build_orchestrator(fail_roles={"qa-model"})

    with pytest.raises(RuntimeError, match="boom:qa-model"):
        async with orchestrator.fanout(["qa"]) as session:
            await session.run_role("qa", [PromptRequest(user_prompt="hi")])

    # The failing client was created and must be torn down.
    assert len(clients) == 1
    assert clients[0].closed is True

    await orchestrator.close()


# --------------------------------------------------------------------------- #
# Lifecycle invariants
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fanout_rejects_calls_after_close() -> None:
    orchestrator, _ = _build_orchestrator()

    async with orchestrator.fanout(["qa"]) as session:
        pass

    with pytest.raises(RuntimeError, match="already closed"):
        await session.run_role("qa", [PromptRequest(user_prompt="too late")])

    await orchestrator.close()


@pytest.mark.asyncio
async def test_fanout_local_provider_swap_closes_resolver_context() -> None:
    # Ensures the swap path tears down the provider_config_resolver context
    # (vLLM container lifecycle), not just the HTTP client object.
    resolver_events: list[tuple[str, str]] = []
    created: list[FakeClient] = []

    async def _noop_close() -> None:
        return None

    @asynccontextmanager
    async def fake_resolver(provider_config: ProviderConfig, *, model: str):
        resolver_events.append(("enter", model))
        resolved = replace(provider_config, api_base=f"http://localhost/{model}")
        yield ResolvedProviderHandle(provider_config=resolved, close=_noop_close)
        resolver_events.append(("exit", model))

    def client_builder(provider_config: ProviderConfig) -> FakeClient:
        c = FakeClient(provider_config)
        created.append(c)
        return c

    orchestrator = ModelOrchestrator(
        _config(provider=Provider.LOCAL),
        recorder=InMemoryRecorder(),
        client_builder=client_builder,
        provider_config_resolver=fake_resolver,
    )

    async with orchestrator.fanout(["qa", "reasoning"]) as session:
        await session.run_role("qa", [PromptRequest(user_prompt="a")])
        await session.run_role("reasoning", [PromptRequest(user_prompt="b")])

    # enter qa, exit qa (swap), enter reasoning, exit reasoning (session close)
    assert resolver_events == [
        ("enter", "qa-model"),
        ("exit", "qa-model"),
        ("enter", "reasoning-model"),
        ("exit", "reasoning-model"),
    ]
    assert all(c.closed for c in created)

    await orchestrator.close()
