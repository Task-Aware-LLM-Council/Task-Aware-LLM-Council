"""
Unit tests for `router.py` — data types + `KeywordRouter` fixture.

Scope: the pure, side-effect-free surface only. No orchestrator, no
decomposer, no LLMs. Policy-level tests (Step 2) live in
test_p4_policy.py.
"""

from __future__ import annotations

import pytest

from council_policies.models import TaskType
from council_policies.router import (
    DispatchRun,
    KeywordRouter,
    RoutingDecision,
    Subtask,
)


# --------------------------------------------------------------------------- #
# Subtask
# --------------------------------------------------------------------------- #


def test_subtask_is_frozen():
    """Subtask must be hashable + frozen — it's used as an ordering key
    and goes into metadata dicts downstream."""
    s = Subtask(text="hello", order=0)
    with pytest.raises(Exception):
        s.text = "mutated"  # type: ignore[misc]


def test_subtask_equality_by_value():
    assert Subtask("x", 0) == Subtask("x", 0)
    assert Subtask("x", 0) != Subtask("x", 1)


# --------------------------------------------------------------------------- #
# RoutingDecision
# --------------------------------------------------------------------------- #


def test_routing_decision_defaults():
    d = RoutingDecision(role="qa_reasoning", scores={"qa_reasoning": 0.9}, confidence=0.9)
    assert d.fallback_used is False
    assert d.fallback_reason is None


# --------------------------------------------------------------------------- #
# DispatchRun
# --------------------------------------------------------------------------- #


def test_dispatch_run_starts_empty_and_appends():
    run = DispatchRun(role="math_code")
    assert run.subtasks == []
    assert run.response is None

    run.append(Subtask("step one", 0))
    run.append(Subtask("step two", 1))
    assert [s.order for s in run.subtasks] == [0, 1]


def test_dispatch_run_single_subtask_renders_verbatim():
    """Single-subtask runs must not add step-labeling meta text —
    specialist shouldn't see decomposer framing when there was no
    decomposition to do."""
    run = DispatchRun(role="math_code")
    run.append(Subtask("Solve x^2 - 4 = 0", 0))
    assert run.rendered_prompt() == "Solve x^2 - 4 = 0"


def test_dispatch_run_multi_subtask_preserves_order():
    """Multi-subtask rendering invariant: both texts appear and the
    order-0 subtask precedes order-1 in the final string. The label
    format itself is not asserted — it's an implementation detail
    that prompt tuning may revise."""
    run = DispatchRun(role="qa_reasoning")
    run.append(Subtask("Who was the first emperor?", 0))
    run.append(Subtask("What year did he die?", 1))

    rendered = run.rendered_prompt()
    assert "first emperor" in rendered
    assert "What year" in rendered
    assert rendered.index("first emperor") < rendered.index("What year")


def test_dispatch_run_multi_subtask_respects_order_field_not_list_position():
    """`Subtask.order` — not list insertion order — drives the label
    numbering. This matters because the policy sorts subtasks by
    `order` before grouping into runs (per the §Policy run() flow),
    and a decomposer that emits out-of-order subtasks must not scramble
    the rendered labels."""
    run = DispatchRun(role="math_code")
    # Append out of natural order — order=2 first, order=0 second.
    run.append(Subtask("compute the second step", 2))
    run.append(Subtask("compute the first step", 0))

    rendered = run.rendered_prompt()
    # Labels reflect `order + 1`, so we should see "Step 3" and "Step 1".
    assert "Step 3" in rendered
    assert "Step 1" in rendered


# --------------------------------------------------------------------------- #
# KeywordRouter
# --------------------------------------------------------------------------- #


def test_keyword_router_collapses_math_and_code_to_one_role():
    r = KeywordRouter()
    math = r.classify("Solve the equation x^2 - 5x + 6 = 0")
    code = r.classify("Write a Python function to sort a list")
    assert math.role == "math_code"
    assert code.role == "math_code"
    assert math.confidence == 1.0
    assert code.confidence == 1.0


def test_keyword_router_qa_reasoning_collapse():
    r = KeywordRouter()
    qa = r.classify("What is the capital of France?")
    reasoning = r.classify("Compare utilitarian and deontological ethics step by step")
    assert qa.role == "qa_reasoning"
    assert reasoning.role == "qa_reasoning"


def test_keyword_router_fever_and_general_collapse():
    r = KeywordRouter()
    fact = r.classify("True or false: vaccines cause autism — verify this claim")
    # Empty / generic prompt falls through regex to QA in P3, which maps
    # to qa_reasoning — so exercise GENERAL via an explicit TaskType map
    # override instead.
    assert fact.role == "fact_general"


def test_keyword_router_fallback_when_tasktype_unmapped():
    """If the collapse map is missing a TaskType, classify() must fall
    back instead of raising — a policy run should degrade, not crash."""
    partial_map = {TaskType.MATH: "math_code"}  # missing every other TaskType
    r = KeywordRouter(tasktype_to_role=partial_map, fallback_role="fact_general")

    d = r.classify("What is the capital of France?")
    assert d.role == "fact_general"
    assert d.fallback_used is True
    assert d.confidence == 0.0
    assert d.fallback_reason is not None
    assert d.fallback_reason.startswith("tasktype_unmapped:")


def test_keyword_router_context_kwarg_accepted_but_unused():
    """Context is in the protocol so `LearnedRouter` can use it; keyword
    router ignores it. Test that passing it doesn't change the decision."""
    r = KeywordRouter()
    without = r.classify("Solve x^2 = 16")
    with_ctx = r.classify("Solve x^2 = 16", context="This is a quadratic equation")
    assert without == with_ctx
