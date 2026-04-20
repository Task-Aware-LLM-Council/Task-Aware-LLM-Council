"""
Tests for `LearnedRouterPolicy` (P4, adapter-shaped).

Coverage matrix:
  * Construction validation (max_subtasks only — role checks are lazy)
  * Lazy role validation in plan():
      - primary + fallback both missing → RuntimeError
      - synthesizer missing but synthesis needed → RuntimeError
  * Passthrough decomposer → short-circuit (no synth phase)
  * 2 subtasks → 2 different roles → 2 runs → 1 synth call
  * 2 adjacent same-role subtasks → 1 run → 1 specialist call
  * 3 subtasks [A, B, A] → 3 runs (non-adjacent same-role do NOT merge)
  * Low-confidence router decision → fallback propagates to metadata
  * max_subtasks cap respected and ordering preserved
  * Phase sequencing: specialist orchestrator fully exits before
    synthesizer enters (GPU-budget invariant)
  * Batch amortizes: one specialist-phase entry + one synth-phase entry
    for N multi-run requests

All tests route through `CouncilBenchmarkRunner` — the policy is now
adapter-shaped and the runner owns the two-phase lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from llm_gateway import PromptRequest

from council_policies import (
    CouncilBenchmarkRunner,
    LearnedRouterPolicy,
    PassthroughDecomposer,
    PolicyRuntime,
    RoutingDecision,
    Subtask,
)

from conftest import ConfigSentinel, FakeClient, FakeOrchestrator  # noqa: E402


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


@dataclass
class StubRouter:
    """Returns the role dictated by `{prompt_substring: role}`. Confidence
    is 1.0 unless `low_confidence_for` substring matches."""

    role_map: dict[str, str]
    default_role: str = "fact_general"
    low_confidence_for: str | None = None

    def classify(self, prompt: str, *, context: str = "") -> RoutingDecision:
        del context
        for needle, role in self.role_map.items():
            if needle in prompt:
                if self.low_confidence_for is not None and self.low_confidence_for in prompt:
                    return RoutingDecision(
                        role=role,
                        scores={role: 0.1},
                        confidence=0.1,
                        fallback_used=True,
                        fallback_reason="low_confidence",
                    )
                return RoutingDecision(role=role, scores={role: 1.0}, confidence=1.0)
        return RoutingDecision(
            role=self.default_role,
            scores={self.default_role: 0.0},
            confidence=0.0,
            fallback_used=True,
            fallback_reason="no_match",
        )


class StubDecomposer:
    """Returns a canned subtask list, ignoring input."""

    def __init__(self, subtasks: list[Subtask]) -> None:
        self._subtasks = subtasks

    async def decompose(self, prompt: str, context: str = "") -> list[Subtask]:
        del prompt, context
        return list(self._subtasks)


# --------------------------------------------------------------------------- #
# Runner builder — reduces boilerplate across tests
# --------------------------------------------------------------------------- #


def _build_runtime(
    specialist_config: ConfigSentinel,
    synthesizer_config: ConfigSentinel,
    orchestrator_factory,
) -> PolicyRuntime:
    return PolicyRuntime(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
    )


def _build_runner(
    *,
    specialist_config: ConfigSentinel,
    synthesizer_config: ConfigSentinel,
    orchestrator_factory,
    router,
    decomposer,
    **policy_kwargs,
) -> CouncilBenchmarkRunner:
    runtime = _build_runtime(specialist_config, synthesizer_config, orchestrator_factory)
    policy = LearnedRouterPolicy(
        router=router,
        decomposer=decomposer,
        **policy_kwargs,
    )
    return CouncilBenchmarkRunner(policies=(policy,), runtime=runtime)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_construction_rejects_zero_max_subtasks():
    with pytest.raises(ValueError, match="max_subtasks"):
        LearnedRouterPolicy(
            router=StubRouter(role_map={}),
            decomposer=PassthroughDecomposer(),
            max_subtasks=0,
        )


# --------------------------------------------------------------------------- #
# Lazy role validation (moved out of __init__ into plan())
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_plan_raises_when_primary_and_fallback_both_missing(
    synthesizer_config, synthesizer_orch,
):
    """Router emits 'math_code'. Specialist config registers neither
    'math_code' nor the fallback 'fact_general' → plan() raises with a
    message naming both roles."""
    specialist_config = ConfigSentinel(("some_other_role",))
    specialist_orch = FakeOrchestrator({"some_other_role": FakeClient("some_other_role")})

    def factory(config):
        if config is specialist_config:
            return specialist_orch
        if config is synthesizer_config:
            return synthesizer_orch
        raise AssertionError(f"unexpected config: {config!r}")

    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=factory,
        router=StubRouter(role_map={"math": "math_code"}),
        decomposer=PassthroughDecomposer(),
    )

    result = (
        await runner.run([PromptRequest(user_prompt="solve this math problem")])
    ).results[0]

    # Runner captures per-policy failures instead of bubbling, so the
    # error surfaces as structured metadata rather than an exception.
    assert result.metrics.error == "RuntimeError"
    assert "math_code" in result.metadata["error_message"]
    assert "fact_general" in result.metadata["error_message"]


@pytest.mark.asyncio
async def test_plan_raises_when_synthesizer_missing_and_synthesis_needed(
    specialist_config, specialist_orch,
):
    """Two different routed roles → 2 runs → synthesis needed. But the
    synthesizer config has no 'synthesizer' role registered → plan()
    raises."""
    synthesizer_config = ConfigSentinel(("wrong_synth_name",))
    synthesizer_orch = FakeOrchestrator(
        {"wrong_synth_name": FakeClient("wrong_synth_name")}
    )

    def factory(config):
        if config is specialist_config:
            return specialist_orch
        if config is synthesizer_config:
            return synthesizer_orch
        raise AssertionError(f"unexpected config: {config!r}")

    subtasks = [
        Subtask("math step", order=0),
        Subtask("qa step", order=1),
    ]
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=factory,
        router=StubRouter(role_map={"math": "math_code", "qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks),
    )

    result = (
        await runner.run([PromptRequest(user_prompt="composite")])
    ).results[0]
    assert result.metrics.error == "RuntimeError"
    assert "synthesizer_role" in result.metadata["error_message"]


@pytest.mark.asyncio
async def test_plan_does_not_check_synthesizer_when_single_run(
    specialist_config, synthesizer_orch, specialist_orch,
):
    """Single-run requests short-circuit synthesis, so a missing
    synthesizer role must NOT raise. This is the dual of the previous
    test and proves the validation is truly lazy."""
    synthesizer_config = ConfigSentinel(("wrong_synth_name",))

    def factory(config):
        if config is specialist_config:
            return specialist_orch
        if config is synthesizer_config:
            return synthesizer_orch  # harmless; never opened
        raise AssertionError(f"unexpected config: {config!r}")

    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=factory,
        router=StubRouter(role_map={"math": "math_code"}),
        decomposer=PassthroughDecomposer(),
    )

    result = await runner.run([PromptRequest(user_prompt="solve math")])
    assert result.results[0].response.text == "response-from-math_code"
    assert synthesizer_orch.enter_count == 0


# --------------------------------------------------------------------------- #
# Single-subtask (passthrough) — P3 parity, no synth phase
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_passthrough_single_run_skips_synth_phase(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"math": "math_code"}),
        decomposer=PassthroughDecomposer(),
    )

    result = await runner.run([PromptRequest(user_prompt="solve this math problem")])

    p4 = result.results[0]
    assert p4.policy_id == "p4"
    assert p4.response.text == "response-from-math_code"

    assert specialist_orch.enter_count == 1
    assert specialist_orch.exit_count == 1
    assert specialist_orch.load_all_calls == [1]
    assert len(specialist_orch.get_client("math_code").requests) == 1

    assert synthesizer_orch.enter_count == 0
    assert synthesizer_orch.exit_count == 0
    assert len(synthesizer_orch.get_client("synthesizer").requests) == 0

    assert p4.metadata["synthesis_short_circuit"] == "single_run"


@pytest.mark.asyncio
async def test_passthrough_forwards_original_prompt_verbatim(
    specialist_config, synthesizer_config, orchestrator_factory, specialist_orch,
):
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"math": "math_code"}),
        decomposer=PassthroughDecomposer(),
    )

    await runner.run([PromptRequest(user_prompt="solve 2+2 math")])

    math_requests = specialist_orch.get_client("math_code").requests
    assert math_requests[0].user_prompt == "solve 2+2 math"


# --------------------------------------------------------------------------- #
# Multi-run — two different roles, both phases fire
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_two_subtasks_two_roles_two_runs_one_synth(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    subtasks = [
        Subtask("compute 2+2 math", order=0),
        Subtask("who was the first emperor qa", order=1),
    ]
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"math": "math_code", "qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks),
    )

    result = await runner.run([PromptRequest(user_prompt="original composite prompt")])
    p4 = result.results[0]

    assert len(specialist_orch.get_client("math_code").requests) == 1
    assert len(specialist_orch.get_client("qa_reasoning").requests) == 1
    assert len(synthesizer_orch.get_client("synthesizer").requests) == 1
    assert p4.response.text == "SYNTHESIZED"

    runs = p4.metadata["runs"]
    assert len(runs) == 2
    assert runs[0]["role"] == "math_code"
    assert runs[1]["role"] == "qa_reasoning"
    assert p4.metadata["synthesis_short_circuit"] is None

    assert specialist_orch.enter_count == 1 and specialist_orch.exit_count == 1
    assert synthesizer_orch.enter_count == 1 and synthesizer_orch.exit_count == 1


# --------------------------------------------------------------------------- #
# Multi-run — adjacent same-role merge
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_two_adjacent_same_role_subtasks_merge_into_one_run(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    subtasks = [
        Subtask("who was the first emperor qa", order=0),
        Subtask("when did he die qa", order=1),
    ]
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks),
    )

    result = await runner.run([PromptRequest(user_prompt="historical questions")])
    p4 = result.results[0]

    qa_requests = specialist_orch.get_client("qa_reasoning").requests
    assert len(qa_requests) == 1
    merged_prompt = qa_requests[0].user_prompt or ""
    assert "first emperor" in merged_prompt
    assert "when did he die" in merged_prompt
    assert merged_prompt.index("first emperor") < merged_prompt.index("when did he die")

    assert p4.metadata["synthesis_short_circuit"] == "single_run"
    assert synthesizer_orch.enter_count == 0

    runs = p4.metadata["runs"]
    assert len(runs) == 1
    assert [s["order"] for s in runs[0]["subtasks"]] == [0, 1]


# --------------------------------------------------------------------------- #
# Multi-run — non-adjacent same-role must NOT merge
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_aba_pattern_produces_three_runs_not_two(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    subtasks = [
        Subtask("qa step one", order=0),
        Subtask("math step two", order=1),
        Subtask("qa step three", order=2),
    ]
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"qa": "qa_reasoning", "math": "math_code"}),
        decomposer=StubDecomposer(subtasks),
    )

    result = await runner.run([PromptRequest(user_prompt="mixed")])
    p4 = result.results[0]

    assert len(specialist_orch.get_client("qa_reasoning").requests) == 2
    assert len(specialist_orch.get_client("math_code").requests) == 1

    runs = p4.metadata["runs"]
    assert len(runs) == 3
    assert [r["role"] for r in runs] == ["qa_reasoning", "math_code", "qa_reasoning"]
    assert [s["order"] for s in runs[0]["subtasks"]] == [0]
    assert [s["order"] for s in runs[1]["subtasks"]] == [1]
    assert [s["order"] for s in runs[2]["subtasks"]] == [2]

    synth_requests = synthesizer_orch.get_client("synthesizer").requests
    assert len(synth_requests) == 1
    synth_prompt = synth_requests[0].user_prompt or ""
    assert (
        synth_prompt.index("Step 1")
        < synth_prompt.index("Step 2")
        < synth_prompt.index("Step 3")
    )


# --------------------------------------------------------------------------- #
# Low-confidence router decision
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_router_fallback_flag_propagates_to_metadata(
    specialist_config, synthesizer_config, orchestrator_factory, specialist_orch,
):
    subtasks = [Subtask("unclear prompt qa", order=0)]
    router = StubRouter(
        role_map={"qa": "qa_reasoning"},
        low_confidence_for="unclear",
    )
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=router,
        decomposer=StubDecomposer(subtasks),
    )

    result = await runner.run([PromptRequest(user_prompt="whatever")])
    p4 = result.results[0]

    assert p4.metadata["router_fallback_used"] == [True]
    assert len(specialist_orch.get_client("qa_reasoning").requests) == 1


# --------------------------------------------------------------------------- #
# max_subtasks cap / ordering
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_max_subtasks_truncates_in_order(
    specialist_config, synthesizer_config, orchestrator_factory,
):
    subtasks = [Subtask(f"subtask qa {i}", order=i) for i in range(5)]
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks),
        max_subtasks=2,
    )

    result = await runner.run([PromptRequest(user_prompt="long")])
    p4 = result.results[0]

    kept = p4.metadata["subtasks"]
    assert len(kept) == 2
    assert [s["order"] for s in kept] == [0, 1]


@pytest.mark.asyncio
async def test_out_of_order_subtasks_are_sorted_before_grouping(
    specialist_config, synthesizer_config, orchestrator_factory,
):
    subtasks = [
        Subtask("qa second", order=1),
        Subtask("math first", order=0),
        Subtask("qa third", order=2),
    ]
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"qa": "qa_reasoning", "math": "math_code"}),
        decomposer=StubDecomposer(subtasks),
    )

    result = await runner.run([PromptRequest(user_prompt="mixed")])
    p4 = result.results[0]

    runs = p4.metadata["runs"]
    assert [r["role"] for r in runs] == ["math_code", "qa_reasoning"]
    assert [s["order"] for s in runs[1]["subtasks"]] == [1, 2]


# --------------------------------------------------------------------------- #
# Phase sequencing — GPU-budget invariant
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_specialist_phase_fully_exits_before_synth_phase_enters(
    specialist_config, synthesizer_config,
    specialist_orch, synthesizer_orch,
):
    """The runner must close the specialist orchestrator before opening
    the synthesizer orchestrator. This is the invariant the reviewer
    flagged on PR #10 — we keep a P4-specific test of it even though the
    runner-level test covers the generic case, because regressions would
    silently double peak GPU residency."""
    events: list[str] = []

    class _Recorder:
        def __init__(self, label: str, inner: FakeOrchestrator) -> None:
            self._label = label
            self._inner = inner

        async def __aenter__(self):
            events.append(f"{self._label}_enter")
            await self._inner.__aenter__()
            return self

        async def __aexit__(self, *args):
            events.append(f"{self._label}_exit")
            return await self._inner.__aexit__(*args)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def recording_factory(config):
        if config is specialist_config:
            return _Recorder("spec", specialist_orch)
        if config is synthesizer_config:
            return _Recorder("synth", synthesizer_orch)
        raise AssertionError(f"unexpected config {config!r}")

    subtasks = [
        Subtask("math step", order=0),
        Subtask("qa step", order=1),
    ]
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=recording_factory,
        router=StubRouter(role_map={"math": "math_code", "qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks),
    )

    await runner.run([PromptRequest(user_prompt="mixed")])

    assert events == ["spec_enter", "spec_exit", "synth_enter", "synth_exit"]


# --------------------------------------------------------------------------- #
# Batch amortization
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_batch_amortizes_phase_entries(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    """Three multi-run requests through a single runner call.
    Specialist orchestrator opens once; synthesizer opens once. Without
    amortization each request would pay its own cold-start."""
    subtasks_template = [
        Subtask("math step", order=0),
        Subtask("qa step", order=1),
    ]
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"math": "math_code", "qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks_template),
    )

    requests = [PromptRequest(user_prompt=f"q{i}") for i in range(3)]
    result = await runner.run(requests)

    assert len(result.results) == 3
    assert specialist_orch.enter_count == 1
    assert specialist_orch.exit_count == 1
    assert synthesizer_orch.enter_count == 1
    assert synthesizer_orch.exit_count == 1
    assert len(specialist_orch.get_client("math_code").requests) == 3
    assert len(specialist_orch.get_client("qa_reasoning").requests) == 3
    assert len(synthesizer_orch.get_client("synthesizer").requests) == 3


@pytest.mark.asyncio
async def test_batch_skips_synth_phase_when_no_multi_run_requests(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    """If every request is single-run, the synthesizer phase must not
    open — saving the cold-start when no fusion is needed."""
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"math": "math_code"}),
        decomposer=PassthroughDecomposer(),
    )

    requests = [PromptRequest(user_prompt=f"math q{i}") for i in range(3)]
    await runner.run(requests)

    assert specialist_orch.enter_count == 1
    assert synthesizer_orch.enter_count == 0


@pytest.mark.asyncio
async def test_empty_request_list_is_noop(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={}),
        decomposer=PassthroughDecomposer(),
    )
    result = await runner.run([])
    assert result.results == ()
    assert specialist_orch.enter_count == 0
    assert synthesizer_orch.enter_count == 0
