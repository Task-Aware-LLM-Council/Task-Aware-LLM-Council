"""
Shared test fixtures for council_policies tests.

Two-phase pattern (reviewer's PR #10 feedback, 2026-04-18):
  - `FakeOrchestrator` implements the async-context-manager + `load_all`
    surface that `PolicyRuntime` opens on behalf of adapter-shaped
    policies (P3/P4). Real orchestrator spins vLLM subprocesses; the
    fake's enter/exit are no-ops.
  - `orchestrator_factory` maps sentinel configs → fake orchestrators,
    so tests can inject specialist and synthesizer fakes separately.

Pattern matches the FakeClient / FakeOrchestrator guidance in
tests/CLAUDE.md — only `get_client(role)` is exercised on the client
path, so that's all the fake implements. `make_orch_response()` builds a
minimal-but-valid `OrchestratorResponse` so tests don't have to
duplicate its many required fields.
"""

from __future__ import annotations

from typing import Any

import pytest
from llm_gateway import PromptRequest, PromptResponse
from model_orchestration import OrchestratorResponse
from model_orchestration.models import OrchestratorCallRecord


# --------------------------------------------------------------------------- #
# OrchestratorResponse factory
# --------------------------------------------------------------------------- #


def make_orch_response(role: str, text: str) -> OrchestratorResponse:
    """Build a minimal valid `OrchestratorResponse` for a fake client."""
    prompt_req = PromptRequest(user_prompt="(placeholder — FakeClient overwrites)")
    prompt_resp = PromptResponse(model=f"{role}-model", text=text)
    call_record = OrchestratorCallRecord(
        event_id=f"evt-{role}",
        started_at="2026-04-18T00:00:00",
        completed_at="2026-04-18T00:00:01",
        target=role,
        resolved_alias=role,
        model=f"{role}-model",
        provider="fake",
        provider_mode="test",
        request={},
        request_metadata={},
    )
    return OrchestratorResponse(
        target=role,
        resolved_alias=role,
        model=f"{role}-model",
        provider="fake",
        provider_mode="test",
        request=prompt_req,
        prompt_response=prompt_resp,
        started_at="2026-04-18T00:00:00",
        completed_at="2026-04-18T00:00:01",
        call_record=call_record,
    )


# --------------------------------------------------------------------------- #
# FakeClient / FakeOrchestrator
# --------------------------------------------------------------------------- #


class FakeClient:
    """Records requests and returns either a canned response text or
    raises when `fail=True`. One client per role."""

    def __init__(
        self,
        role: str,
        *,
        text: str | None = None,
        fail: bool = False,
        response_fn: Any = None,
    ) -> None:
        self.role = role
        self.text = text if text is not None else f"response-from-{role}"
        self.fail = fail
        self.response_fn = response_fn  # optional: callable(request) -> str
        self.requests: list[PromptRequest] = []

    async def get_response(self, request: PromptRequest) -> OrchestratorResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError(f"boom:{self.role}")
        text = self.response_fn(request) if self.response_fn else self.text
        return make_orch_response(self.role, text)


class FakeOrchestrator:
    """Async-context-manager + `load_all` + `get_client`. Real
    `ModelOrchestrator` launches vLLM subprocesses on enter and reclaims
    GPU memory on exit — the fake's enter/exit are pure bookkeeping so
    tests can assert the policy opened/closed it correctly."""

    def __init__(self, clients: dict[str, FakeClient]) -> None:
        self._clients = clients
        self.enter_count = 0
        self.exit_count = 0
        self.load_all_calls: list[int | None] = []

    def get_client(self, role: str) -> FakeClient:
        if role not in self._clients:
            raise KeyError(role)
        return self._clients[role]

    async def load_all(
        self,
        targets: tuple[str, ...] | None = None,
        *,
        max_parallel: int | None = None,
    ) -> None:
        del targets
        self.load_all_calls.append(max_parallel)

    async def __aenter__(self) -> FakeOrchestrator:
        self.enter_count += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exit_count += 1


# --------------------------------------------------------------------------- #
# Config sentinels + orchestrator factory
# --------------------------------------------------------------------------- #
#
# P4 takes `specialist_config` and `synthesizer_config` (OrchestratorConfig
# objects in prod). Tests don't care about the config internals — they
# care that the factory dispatches each config to the right fake. So we
# use a minimal ConfigSentinel with a `models` tuple that the policy's
# construction-time role-registration check walks.


class _RoleSpec:
    """Minimal ModelSpec stand-in: just the attributes that
    `PolicyRuntime._known_roles` walks."""
    def __init__(self, role: str, aliases: tuple[str, ...] = ()) -> None:
        self.role = role
        self.aliases = aliases


class ConfigSentinel:
    """Stand-in for `OrchestratorConfig` during tests. Exposes `models`
    so `PolicyRuntime._known_roles` can enumerate registered
    roles/aliases without spinning up a real orchestrator."""
    def __init__(self, roles: tuple[str, ...]) -> None:
        self.models = tuple(_RoleSpec(r, (r,)) for r in roles)


# --------------------------------------------------------------------------- #
# Common fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def specialist_orch() -> FakeOrchestrator:
    return FakeOrchestrator(
        {
            "math_code": FakeClient("math_code"),
            "qa_reasoning": FakeClient("qa_reasoning"),
            "fact_general": FakeClient("fact_general"),
        }
    )


@pytest.fixture
def synthesizer_orch() -> FakeOrchestrator:
    return FakeOrchestrator(
        {"synthesizer": FakeClient("synthesizer", text="SYNTHESIZED")}
    )


@pytest.fixture
def specialist_config() -> ConfigSentinel:
    return ConfigSentinel(("math_code", "qa_reasoning", "fact_general"))


@pytest.fixture
def synthesizer_config() -> ConfigSentinel:
    return ConfigSentinel(("synthesizer",))


@pytest.fixture
def orchestrator_factory(
    specialist_orch: FakeOrchestrator,
    synthesizer_orch: FakeOrchestrator,
    specialist_config: ConfigSentinel,
    synthesizer_config: ConfigSentinel,
):
    """Returns a callable the policy uses in place of `ModelOrchestrator`:
    maps config identity → the right fake. Any unexpected config is a bug
    in the test, so we raise loudly."""
    def _factory(config: Any) -> FakeOrchestrator:
        if config is specialist_config:
            return specialist_orch
        if config is synthesizer_config:
            return synthesizer_orch
        raise AssertionError(f"unexpected config passed to factory: {config!r}")
    return _factory
