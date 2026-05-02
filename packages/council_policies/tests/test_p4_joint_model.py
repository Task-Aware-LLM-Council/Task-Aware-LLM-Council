"""
Tests for `LearnedRouterPolicy(use_joint_model=True)` — the branch
where the decomposer (`Seq2SeqDecomposerRouter`) emits routing
decisions via `Subtask.suggested_role` and the policy does NOT call a
separate `Router`.

Coverage matrix:
  * Construction validation (flag + router/decomposer combinations)
  * Dispatch parity with the two-component stack when the joint model
    emits canonical roles
  * Missing role → fallback with `joint_model_missing_role`
  * Unknown role → fallback with `joint_model_unknown_role`
  * metadata["router_type"] == "Seq2SeqDecomposerRouter"
  * Single-run short-circuit still works (no synth call)

The joint model is injected as a `Seq2SeqDecomposerRouter` with a
`FakeGenerate` — same pattern as test_seq2seq_decomposer_router.py.
Routing through the `FakeOrchestrator` pattern in tests/conftest.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from llm_gateway import PromptRequest

from council_policies.p4 import (
    CouncilBenchmarkRunner,
    LearnedRouterPolicy,
    PassthroughDecomposer,
    PolicyRuntime,
    Seq2SeqDecomposerRouter,
)

from conftest import FakeClient, FakeOrchestrator  # noqa: E402


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


@dataclass
class FakeGenerate:
    text: str = "[]"
    fail: bool = False
    last_input: str | None = field(default=None)

    def __call__(self, text: str) -> str:
        self.last_input = text
        if self.fail:
            raise RuntimeError("boom")
        return self.text


def _json(items: list[dict]) -> str:
    return json.dumps(items)


def _joint_decomposer(items: list[dict]) -> Seq2SeqDecomposerRouter:
    return Seq2SeqDecomposerRouter(generate_fn=FakeGenerate(text=_json(items)))


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_joint_flag_rejects_router_param():
    with pytest.raises(ValueError, match="router.*must be None"):
        LearnedRouterPolicy(
            router=object(),  # type: ignore[arg-type]
            decomposer=_joint_decomposer([]),
            use_joint_model=True,
        )


def test_joint_flag_requires_decomposer():
    with pytest.raises(ValueError, match="decomposer.*required"):
        LearnedRouterPolicy(
            router=None,
            decomposer=None,
            use_joint_model=True,
        )


def test_non_joint_flag_requires_router():
    with pytest.raises(ValueError, match="router.*required"):
        LearnedRouterPolicy(
            router=None,
            decomposer=PassthroughDecomposer(),
            use_joint_model=False,
        )


# --------------------------------------------------------------------------- #
# Runner builder
# --------------------------------------------------------------------------- #


def _build_runner(
    *,
    specialist_config,
    synthesizer_config,
    orchestrator_factory,
    decomposer,
    **policy_kwargs,
) -> CouncilBenchmarkRunner:
    runtime = PolicyRuntime(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
    )
    policy = LearnedRouterPolicy(
        router=None,
        decomposer=decomposer,
        use_joint_model=True,
        **policy_kwargs,
    )
    return CouncilBenchmarkRunner(policies=(policy,), runtime=runtime)


# --------------------------------------------------------------------------- #
# Happy path — joint model emits canonical roles
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_joint_two_roles_produce_two_runs_and_synth(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    decomposer = _joint_decomposer(
        [
            {"role": "math_code", "subtask": "compute 2+2"},
            {"role": "qa_reasoning", "subtask": "explain the result"},
        ]
    )
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        decomposer=decomposer,
    )

    result = await runner.run([PromptRequest(user_prompt="composite")])
    p4 = result.results[0]

    assert len(specialist_orch.get_client("math_code").requests) == 1
    assert len(specialist_orch.get_client("qa_reasoning").requests) == 1
    assert len(synthesizer_orch.get_client("synthesizer").requests) == 1
    assert p4.response.text == "SYNTHESIZED"

    runs = p4.metadata["runs"]
    assert [r["role"] for r in runs] == ["math_code", "qa_reasoning"]
    assert p4.metadata["router_type"] == "Seq2SeqDecomposerRouter"
    assert p4.metadata["router_fallback_used"] == [False, False]
    assert p4.metadata["router_confidence"] == [1.0, 1.0]


@pytest.mark.asyncio
async def test_joint_adjacent_same_role_subtasks_merge(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    """Same dispatch-grouping invariant as the two-component stack:
    adjacent same-role subtasks collapse into one specialist call, and
    a single resulting run short-circuits synthesis."""
    decomposer = _joint_decomposer(
        [
            {"role": "qa_reasoning", "subtask": "first question"},
            {"role": "qa_reasoning", "subtask": "follow-up question"},
        ]
    )
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        decomposer=decomposer,
    )

    result = await runner.run([PromptRequest(user_prompt="two qa questions")])
    p4 = result.results[0]

    qa_requests = specialist_orch.get_client("qa_reasoning").requests
    assert len(qa_requests) == 1
    merged = qa_requests[0].user_prompt or ""
    assert "first question" in merged
    assert "follow-up question" in merged

    assert p4.metadata["synthesis_short_circuit"] == "single_run"
    assert synthesizer_orch.enter_count == 0


# --------------------------------------------------------------------------- #
# Fallback — joint model emits missing or unknown role
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_joint_missing_role_routes_to_fallback(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch,
):
    decomposer = _joint_decomposer(
        [{"subtask": "no role field here"}]  # role key omitted
    )
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        decomposer=decomposer,
    )

    result = await runner.run([PromptRequest(user_prompt="q")])
    p4 = result.results[0]

    # Default fallback is `fact_general`.
    assert len(specialist_orch.get_client("fact_general").requests) == 1
    assert p4.metadata["router_fallback_used"] == [True]
    assert p4.metadata["router_confidence"] == [0.0]
    runs = p4.metadata["runs"]
    assert runs[0]["role"] == "fact_general"


@pytest.mark.asyncio
async def test_joint_unknown_role_routes_to_fallback(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch,
):
    """`math-code` (hyphen typo) is NOT in ROLE_LABELS → policy stamps
    `joint_model_unknown_role` and routes through the fallback role."""
    decomposer = _joint_decomposer(
        [{"role": "math-code", "subtask": "compute something"}]
    )
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        decomposer=decomposer,
    )

    result = await runner.run([PromptRequest(user_prompt="typo test")])
    p4 = result.results[0]

    assert len(specialist_orch.get_client("fact_general").requests) == 1
    assert len(specialist_orch.get_client("math_code").requests) == 0
    assert p4.metadata["router_fallback_used"] == [True]


@pytest.mark.asyncio
async def test_joint_unregistered_role_routes_to_fallback_via_resolve_role():
    """Valid role vocab but not registered on the specialist config →
    `_resolve_role` kicks in and sends it to fallback. Catches the case
    where the joint model is correct but specialist wiring is missing a
    role."""
    from conftest import ConfigSentinel

    specialist_config = ConfigSentinel(("fact_general",))  # no math_code
    specialist_orch = FakeOrchestrator(
        {"fact_general": FakeClient("fact_general")}
    )
    synthesizer_config = ConfigSentinel(("synthesizer",))
    synthesizer_orch = FakeOrchestrator(
        {"synthesizer": FakeClient("synthesizer", text="SYNTHESIZED")}
    )

    def factory(config):
        if config is specialist_config:
            return specialist_orch
        if config is synthesizer_config:
            return synthesizer_orch
        raise AssertionError(f"unexpected config: {config!r}")

    decomposer = _joint_decomposer(
        [{"role": "math_code", "subtask": "compute"}]
    )
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=factory,
        decomposer=decomposer,
    )

    result = await runner.run([PromptRequest(user_prompt="q")])
    p4 = result.results[0]

    assert len(specialist_orch.get_client("fact_general").requests) == 1
    # fallback_used OR'd with "role == self.fallback_role" in the outer
    # loop, so even though _decision_from_joint_model returned
    # fallback_used=False, the resolved role is the fallback → flag flips.
    assert p4.metadata["router_fallback_used"] == [True]


# --------------------------------------------------------------------------- #
# Single-run short-circuit — no synthesizer
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_joint_single_run_skips_synth_phase(
    specialist_config, synthesizer_config, orchestrator_factory,
    specialist_orch, synthesizer_orch,
):
    decomposer = _joint_decomposer(
        [{"role": "math_code", "subtask": "solve 2+2"}]
    )
    runner = _build_runner(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
        orchestrator_factory=orchestrator_factory,
        decomposer=decomposer,
    )

    result = await runner.run([PromptRequest(user_prompt="solve math")])
    p4 = result.results[0]

    assert p4.response.text == "response-from-math_code"
    assert p4.metadata["synthesis_short_circuit"] == "single_run"
    assert synthesizer_orch.enter_count == 0
    assert len(synthesizer_orch.get_client("synthesizer").requests) == 0
    assert p4.metadata["router_type"] == "Seq2SeqDecomposerRouter"
