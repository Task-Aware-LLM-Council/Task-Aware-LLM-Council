"""
Tests for `LearnedRouterPolicy` (P4, two-phase GPU lifecycle).

Coverage matrix:
  * Construction validation (config inspection, not orchestrator probing)
  * Passthrough decomposer + any router → short-circuit (no synth phase)
  * 2 subtasks → 2 different roles → 2 runs → 1 synth call
  * 2 adjacent same-role subtasks → 1 run → 1 specialist call
  * 3 subtasks [A, B, A] → 3 runs (non-adjacent A's do NOT merge)
  * Low-confidence router decision → fallback role dispatched
  * max_subtasks cap respected
  * Phase sequencing: specialists orchestrator fully exits before
    synthesizer enters (the reviewer's core invariant)
  * `run_batch` amortizes: one specialist-phase entry + one synth-phase
    entry for N multi-run requests

Stubs: `StubRouter` (explicit role-per-prompt map) and `StubDecomposer`
(canned subtask list) keep tests deterministic. The orchestrators are
`FakeOrchestrator`s from conftest, wired through `orchestrator_factory`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from llm_gateway import PromptRequest

from council_policies import (
    LearnedRouterPolicy,
    PassthroughDecomposer,
    RoutingDecision,
    Subtask,
)

from conftest import ConfigSentinel, FakeOrchestrator  # noqa: E402 — pytest-added path


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
# Policy builder — reduces boilerplate across tests
# --------------------------------------------------------------------------- #


def _build_policy(
    *,
    specialist_config,
    synthesizer_config,
    orchestrator_factory,
    router,
    decomposer,
    **kwargs,
) -> LearnedRouterPolicy:
    return LearnedRouterPolicy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        router=router,
        decomposer=decomposer,
        orchestrator_factory=orchestrator_factory,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_construction_validates_fallback_role(specialist_config, synthesizer_config):
    with pytest.raises(ValueError, match="fallback_role"):
        LearnedRouterPolicy(
            specialist_config=specialist_config,
            synthesizer_config=synthesizer_config,
            router=StubRouter(role_map={}),
            decomposer=PassthroughDecomposer(),
            fallback_role="nonexistent",
        )


def test_construction_validates_synthesizer_role(specialist_config, synthesizer_config):
    with pytest.raises(ValueError, match="synthesizer_role"):
        LearnedRouterPolicy(
            specialist_config=specialist_config,
            synthesizer_config=synthesizer_config,
            router=StubRouter(role_map={}),
            decomposer=PassthroughDecomposer(),
            synthesizer_role="nonexistent",
        )


def test_construction_rejects_zero_max_subtasks(specialist_config, synthesizer_config):
    with pytest.raises(ValueError, match="max_subtasks"):
        LearnedRouterPolicy(
            specialist_config=specialist_config,
            synthesizer_config=synthesizer_config,
            router=StubRouter(role_map={}),
            decomposer=PassthroughDecomposer(),
            max_subtasks=0,
        )


def test_construction_rejects_zero_load_parallel(specialist_config, synthesizer_config):
    with pytest.raises(ValueError, match="specialist_load_parallel"):
        LearnedRouterPolicy(
            specialist_config=specialist_config,
            synthesizer_config=synthesizer_config,
            router=StubRouter(role_map={}),
            decomposer=PassthroughDecomposer(),
            specialist_load_parallel=0,
        )


# --------------------------------------------------------------------------- #
# Single-subtask (passthrough) — P3 parity, no synth phase at all
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_passthrough_single_run_skips_synth_phase(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    """PassthroughDecomposer → one subtask → one run → Phase 2 never
    runs. The synthesizer orchestrator must NOT be entered."""
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"math": "math_code"}),
        decomposer=PassthroughDecomposer(),
    )

    response = await policy.run_council(PromptRequest(user_prompt="solve this math problem"))

    assert response.policy == "p4"
    assert response.winner.text == "response-from-math_code"

    # Specialist phase ran.
    assert specialist_orch.enter_count == 1
    assert specialist_orch.exit_count == 1
    assert specialist_orch.load_all_calls == [1]  # specialist_load_parallel default
    assert len(specialist_orch.get_client("math_code").requests) == 1

    # Synthesizer phase never started — core short-circuit invariant.
    assert synthesizer_orch.enter_count == 0
    assert synthesizer_orch.exit_count == 0
    assert len(synthesizer_orch.get_client("synthesizer").requests) == 0

    assert response.metadata["synthesis_short_circuit"] == "single_run"


@pytest.mark.asyncio
async def test_passthrough_forwards_original_prompt_verbatim(
    specialist_config, synthesizer_config, orchestrator_factory, specialist_orch,
):
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"math": "math_code"}),
        decomposer=PassthroughDecomposer(),
    )

    await policy.run(PromptRequest(user_prompt="solve 2+2 math"))

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
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"math": "math_code", "qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks),
    )

    response = await policy.run_council(PromptRequest(user_prompt="original composite prompt"))

    assert len(specialist_orch.get_client("math_code").requests) == 1
    assert len(specialist_orch.get_client("qa_reasoning").requests) == 1
    assert len(synthesizer_orch.get_client("synthesizer").requests) == 1
    assert response.winner.text == "SYNTHESIZED"

    # Two runs, each with one subtask.
    assert len(response.metadata["runs"]) == 2
    assert response.metadata["runs"][0]["role"] == "math_code"
    assert response.metadata["runs"][1]["role"] == "qa_reasoning"
    assert response.metadata["synthesis_short_circuit"] is None

    # Both phases fired exactly once.
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
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks),
    )

    response = await policy.run_council(PromptRequest(user_prompt="historical questions"))

    qa_requests = specialist_orch.get_client("qa_reasoning").requests
    assert len(qa_requests) == 1
    merged_prompt = qa_requests[0].user_prompt or ""
    assert "first emperor" in merged_prompt
    assert "when did he die" in merged_prompt
    assert merged_prompt.index("first emperor") < merged_prompt.index("when did he die")

    # One run → no synth phase.
    assert response.metadata["synthesis_short_circuit"] == "single_run"
    assert synthesizer_orch.enter_count == 0
    assert len(response.metadata["runs"]) == 1
    assert response.metadata["runs"][0]["subtask_orders"] == [0, 1]


# --------------------------------------------------------------------------- #
# Multi-run — non-adjacent same-role must NOT merge (ordering invariant)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_aba_pattern_produces_three_runs_not_two(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    """Core ordering invariant: [qa, math, qa] must produce 3 runs, not
    2. Merging the two qa's would scramble the synthesizer's view."""
    subtasks = [
        Subtask("qa step one", order=0),
        Subtask("math step two", order=1),
        Subtask("qa step three", order=2),
    ]
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"qa": "qa_reasoning", "math": "math_code"}),
        decomposer=StubDecomposer(subtasks),
    )

    response = await policy.run_council(PromptRequest(user_prompt="mixed"))

    assert len(specialist_orch.get_client("qa_reasoning").requests) == 2
    assert len(specialist_orch.get_client("math_code").requests) == 1

    runs_meta = response.metadata["runs"]
    assert len(runs_meta) == 3
    assert [r["role"] for r in runs_meta] == ["qa_reasoning", "math_code", "qa_reasoning"]
    assert runs_meta[0]["subtask_orders"] == [0]
    assert runs_meta[1]["subtask_orders"] == [1]
    assert runs_meta[2]["subtask_orders"] == [2]

    synth_requests = synthesizer_orch.get_client("synthesizer").requests
    assert len(synth_requests) == 1
    synth_prompt = synth_requests[0].user_prompt or ""
    assert synth_prompt.index("Step 1") < synth_prompt.index("Step 2") < synth_prompt.index("Step 3")


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
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=router,
        decomposer=StubDecomposer(subtasks),
    )

    response = await policy.run_council(PromptRequest(user_prompt="whatever"))

    assert response.metadata["router_fallback_used"] == [True]
    assert len(specialist_orch.get_client("qa_reasoning").requests) == 1


# --------------------------------------------------------------------------- #
# max_subtasks cap / ordering
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_max_subtasks_truncates_in_order(
    specialist_config, synthesizer_config, orchestrator_factory,
):
    subtasks = [Subtask(f"subtask qa {i}", order=i) for i in range(5)]
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks),
        max_subtasks=2,
    )

    response = await policy.run_council(PromptRequest(user_prompt="long"))

    kept = response.metadata["subtasks"]
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
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"qa": "qa_reasoning", "math": "math_code"}),
        decomposer=StubDecomposer(subtasks),
    )

    response = await policy.run_council(PromptRequest(user_prompt="mixed"))

    runs_meta = response.metadata["runs"]
    assert [r["role"] for r in runs_meta] == ["math_code", "qa_reasoning"]
    assert runs_meta[1]["subtask_orders"] == [1, 2]


# --------------------------------------------------------------------------- #
# Phase sequencing — the reviewer's core invariant
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_specialist_phase_fully_exits_before_synth_phase_enters(
    specialist_config, synthesizer_config,
    specialist_orch, synthesizer_orch,
):
    """Record the order of aenter/aexit across both phases and assert the
    specialist orchestrator is fully torn down before the synthesizer
    orchestrator enters. This is the GPU-budget invariant the reviewer
    flagged on PR #10.

    Python looks up `__aenter__` on the class, not the instance, so we
    can't monkey-patch the fake — we wrap each orchestrator in a thin
    recorder that proxies attribute access to the underlying fake but
    logs enter/exit on its own class."""
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
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=recording_factory,
        router=StubRouter(role_map={"math": "math_code", "qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks),
    )

    await policy.run(PromptRequest(user_prompt="mixed"))

    assert events == ["spec_enter", "spec_exit", "synth_enter", "synth_exit"]


# --------------------------------------------------------------------------- #
# Batch amortization
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_batch_amortizes_phase_entries(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    """Three requests, all multi-run. Specialist orchestrator should be
    entered once; synthesizer once. Without amortization each request
    would pay its own cold-start."""
    subtasks_template = [
        Subtask("math step", order=0),
        Subtask("qa step", order=1),
    ]
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"math": "math_code", "qa": "qa_reasoning"}),
        decomposer=StubDecomposer(subtasks_template),
    )

    requests = [PromptRequest(user_prompt=f"q{i}") for i in range(3)]
    responses = await policy.run_batch(requests)

    assert len(responses) == 3
    # Both orchestrators entered exactly once across the whole batch.
    assert specialist_orch.enter_count == 1
    assert specialist_orch.exit_count == 1
    assert synthesizer_orch.enter_count == 1
    assert synthesizer_orch.exit_count == 1
    # One specialist dispatch per subtask per request: 2 roles × 3 requests.
    assert len(specialist_orch.get_client("math_code").requests) == 3
    assert len(specialist_orch.get_client("qa_reasoning").requests) == 3
    # One synth call per multi-run request.
    assert len(synthesizer_orch.get_client("synthesizer").requests) == 3


@pytest.mark.asyncio
async def test_run_batch_skips_synth_phase_when_no_multi_run_requests(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    """If every request is single-run, the synthesizer phase must not
    open at all — saving the cold-start when no fusion is needed."""
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={"math": "math_code"}),
        decomposer=PassthroughDecomposer(),
    )

    requests = [PromptRequest(user_prompt=f"math q{i}") for i in range(3)]
    await policy.run_batch(requests)

    assert specialist_orch.enter_count == 1
    assert synthesizer_orch.enter_count == 0


@pytest.mark.asyncio
async def test_run_batch_empty_request_list_is_noop(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    policy = _build_policy(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        router=StubRouter(role_map={}),
        decomposer=PassthroughDecomposer(),
    )
    assert await policy.run_batch([]) == []
    assert specialist_orch.enter_count == 0
    assert synthesizer_orch.enter_count == 0
