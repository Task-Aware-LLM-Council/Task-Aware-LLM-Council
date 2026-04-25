"""
Tests for `PolicyMetrics` population + runner error capture.

Scope: every policy (P3/P4/P2) must surface a populated
`PolicyResult.metrics` after a successful run, and the runner must turn
per-policy failures into `metrics.error` rather than propagating the
exception. Shape-level — exact token counts are delegated to the
provider adapters, not to the council.
"""
from __future__ import annotations

import pytest
from llm_gateway import PromptRequest, PromptResponse, Usage
from model_orchestration import OrchestratorResponse
from model_orchestration.models import OrchestratorCallRecord

from council_policies import (
    CouncilBenchmarkRunner,
    LearnedRouterPolicy,
    P2PromptAdapter,
    P3Adapter,
    PolicyMetrics,
    PolicyRuntime,
    Subtask,
)
from council_policies.prompts import TIEBREAK_SYSTEM_PROMPT, VOTER_SYSTEM_PROMPT

from conftest import ConfigSentinel, FakeClient, FakeOrchestrator


def _response_with_metrics(
    role: str,
    text: str,
    *,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
) -> OrchestratorResponse:
    """Build an OrchestratorResponse with non-zero latency/usage so
    metrics assertions can verify aggregation, not just the happy path
    where everything is zero."""
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    prompt_resp = PromptResponse(
        model=f"{role}-model", text=text, usage=usage, latency_ms=latency_ms
    )
    call_record = OrchestratorCallRecord(
        event_id=f"evt-{role}",
        started_at="2026-04-19T00:00:00",
        completed_at="2026-04-19T00:00:01",
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
        request=PromptRequest(user_prompt="(placeholder)"),
        prompt_response=prompt_resp,
        started_at="2026-04-19T00:00:00",
        completed_at="2026-04-19T00:00:01",
        call_record=call_record,
    )


class MetricClient:
    """FakeClient variant whose response carries latency + usage."""

    def __init__(
        self,
        role: str,
        text: str,
        *,
        latency_ms: float = 100.0,
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> None:
        self.role = role
        self.text = text
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.requests: list[PromptRequest] = []

    async def get_response(self, request: PromptRequest) -> OrchestratorResponse:
        self.requests.append(request)
        return _response_with_metrics(
            self.role,
            self.text,
            latency_ms=self.latency_ms,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


class StubRouter:
    def __init__(self, role_map: dict[str, str]) -> None:
        self._role_map = role_map

    def classify(self, prompt: str, *, context: str = ""):
        del context
        from council_policies.router import RoutingDecision

        for needle, role in self._role_map.items():
            if needle in prompt:
                return RoutingDecision(
                    role=role, scores={role: 0.9}, confidence=0.9
                )
        return RoutingDecision(
            role="fact_general",
            scores={"fact_general": 0.0},
            confidence=0.0,
            fallback_used=True,
        )


class StubDecomposer:
    def __init__(self, subtasks: list[Subtask]) -> None:
        self._subtasks = subtasks

    async def decompose(self, prompt: str, context: str = "") -> list[Subtask]:
        del prompt, context
        return list(self._subtasks)


def _runtime(specialist_orch, synthesizer_orch, spec_cfg, synth_cfg):
    def factory(config):
        if config is spec_cfg:
            return specialist_orch
        if config is synth_cfg:
            return synthesizer_orch
        raise AssertionError(f"unexpected config: {config!r}")

    return PolicyRuntime(
        specialist_config=spec_cfg,
        synthesizer_config=synth_cfg,
        orchestrator_factory=factory,
    )


# --------------------------------------------------------------------------- #
# P3 metrics
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_p3_populates_metrics_on_direct_route():
    spec_cfg = ConfigSentinel(("qa", "general"))
    synth_cfg = ConfigSentinel(("synthesizer",))
    specialist_orch = FakeOrchestrator(
        {"qa": MetricClient("qa", "answer", latency_ms=150.0,
                            input_tokens=20, output_tokens=10)}
    )
    synthesizer_orch = FakeOrchestrator({"synthesizer": FakeClient("synthesizer")})
    runner = CouncilBenchmarkRunner(
        policies=(P3Adapter(),),
        runtime=_runtime(specialist_orch, synthesizer_orch, spec_cfg, synth_cfg),
    )

    (result,) = (
        await runner.run([PromptRequest(user_prompt="What is France?")])
    ).results

    assert isinstance(result.metrics, PolicyMetrics)
    assert result.metrics.error is None
    assert result.metrics.confidence_score == 1.0
    assert result.metrics.latency_ms == 150.0
    assert result.metrics.specialist_latency_ms == {"qa": 150.0}
    assert result.metrics.token_usage.total_tokens == 30
    assert result.metrics.specialist_token_usage["qa"].input_tokens == 20


@pytest.mark.asyncio
async def test_p3_confidence_zero_on_fallback():
    """Primary role missing → fallback fires → confidence_score==0.0."""
    spec_cfg = ConfigSentinel(("general",))
    synth_cfg = ConfigSentinel(("synthesizer",))
    specialist_orch = FakeOrchestrator(
        {"general": MetricClient("general", "fallback-answer")}
    )
    synthesizer_orch = FakeOrchestrator({"synthesizer": FakeClient("synthesizer")})
    runner = CouncilBenchmarkRunner(
        policies=(P3Adapter(),),
        runtime=_runtime(specialist_orch, synthesizer_orch, spec_cfg, synth_cfg),
    )

    # Math prompt routes to "reasoning" which isn't registered → fallback.
    (result,) = (
        await runner.run([PromptRequest(user_prompt="solve x^2 - 4 = 0")])
    ).results
    assert result.metrics.confidence_score == 0.0
    assert result.metadata["router_fallback_used"] is True


# --------------------------------------------------------------------------- #
# P4 metrics
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_p4_aggregates_specialist_and_synthesizer_metrics():
    spec_cfg = ConfigSentinel(("math_code", "qa_reasoning"))
    synth_cfg = ConfigSentinel(("synthesizer",))
    specialist_orch = FakeOrchestrator(
        {
            "math_code": MetricClient(
                "math_code", "math partial", latency_ms=100.0,
                input_tokens=15, output_tokens=5,
            ),
            "qa_reasoning": MetricClient(
                "qa_reasoning", "qa partial", latency_ms=200.0,
                input_tokens=25, output_tokens=8,
            ),
        }
    )
    synthesizer_orch = FakeOrchestrator(
        {
            "synthesizer": MetricClient(
                "synthesizer", "SYNTH", latency_ms=50.0,
                input_tokens=40, output_tokens=20,
            )
        }
    )
    runner = CouncilBenchmarkRunner(
        policies=(
            LearnedRouterPolicy(
                router=StubRouter({"math": "math_code", "qa": "qa_reasoning"}),
                decomposer=StubDecomposer(
                    [Subtask("math step", 0), Subtask("qa step", 1)]
                ),
            ),
        ),
        runtime=_runtime(specialist_orch, synthesizer_orch, spec_cfg, synth_cfg),
    )

    (result,) = (
        await runner.run([PromptRequest(user_prompt="math then qa")])
    ).results

    # Specialist + synthesizer latencies sum. Mean router confidence (0.9)
    # because both subtasks hit a routed role (not fallback).
    assert result.metrics.latency_ms == 350.0
    assert result.metrics.specialist_latency_ms == {
        "math_code": 100.0,
        "qa_reasoning": 200.0,
        "synthesizer": 50.0,
    }
    assert result.metrics.confidence_score == pytest.approx(0.9)
    assert result.metrics.token_usage.total_tokens == (
        (15 + 5) + (25 + 8) + (40 + 20)
    )


# --------------------------------------------------------------------------- #
# P2 metrics
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_p2_metrics_include_voters_and_confidence_margin():
    spec_cfg = ConfigSentinel(("qa", "reasoning", "general"))
    synth_cfg = ConfigSentinel(("synthesizer",))

    def vote_response(request):
        if request.system_prompt in (VOTER_SYSTEM_PROMPT, TIEBREAK_SYSTEM_PROMPT):
            return "A"  # every voter picks the first candidate
        return f"candidate-from-{request.user_prompt[:4]}"

    clients = {}
    for role in ("qa", "reasoning", "general"):
        clients[role] = _MetricClientWithFn(
            role, vote_response, latency_ms=80.0,
            input_tokens=12, output_tokens=6,
        )
    specialist_orch = FakeOrchestrator(clients)
    synthesizer_orch = FakeOrchestrator({"synthesizer": FakeClient("synthesizer")})
    runner = CouncilBenchmarkRunner(
        policies=(P2PromptAdapter(),),
        runtime=_runtime(specialist_orch, synthesizer_orch, spec_cfg, synth_cfg),
    )

    (result,) = (
        await runner.run([PromptRequest(user_prompt="Q?")])
    ).results

    # 3 candidates + 3 voters, each 80ms — total 480ms.
    assert result.metrics.latency_ms == 480.0
    assert set(result.metrics.specialist_latency_ms) == {
        "qa", "reasoning", "general", "_voters"
    }
    # Unanimous vote for first candidate → margin 3/3 = 1.0.
    assert result.metrics.confidence_score == pytest.approx(1.0)
    assert result.metrics.error is None


# --------------------------------------------------------------------------- #
# Runner error capture
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_runner_isolates_failure_to_single_policy():
    """One policy raising in plan() must not sink the batch."""
    spec_cfg = ConfigSentinel(("qa", "general"))
    synth_cfg = ConfigSentinel(("synthesizer",))
    specialist_orch = FakeOrchestrator(
        {"qa": MetricClient("qa", "p3-answer"), "general": MetricClient("general", "x")}
    )
    synthesizer_orch = FakeOrchestrator({"synthesizer": FakeClient("synthesizer")})

    # LearnedRouterPolicy validates role
    # registration in plan() — so a router role and fallback both missing
    # from the specialist config raises *before* any specialist is called.
    runner = CouncilBenchmarkRunner(
        policies=(
            P3Adapter(),
            LearnedRouterPolicy(
                router=StubRouter({"math": "nowhere"}),
                decomposer=StubDecomposer([Subtask("math step", 0)]),
                fallback_role="also_nowhere",
            ),
        ),
        runtime=_runtime(specialist_orch, synthesizer_orch, spec_cfg, synth_cfg),
    )

    results = (
        await runner.run([PromptRequest(user_prompt="What is France?")])
    ).results
    by_policy = {r.policy_id: r for r in results}

    assert by_policy["p3"].metrics.error is None
    assert by_policy["p3"].response.text == "p3-answer"
    assert by_policy["p4"].metrics.error == "RuntimeError"
    assert by_policy["p4"].response.text == ""


class _MetricClientWithFn(MetricClient):
    """MetricClient that picks response text via callable(request)."""

    def __init__(self, role, response_fn, **kwargs):
        super().__init__(role, text="unused", **kwargs)
        self._fn = response_fn

    async def get_response(self, request):
        self.requests.append(request)
        return _response_with_metrics(
            self.role,
            self._fn(request),
            latency_ms=self.latency_ms,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
