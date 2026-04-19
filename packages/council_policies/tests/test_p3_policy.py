"""
Tests for RuleBasedRoutingPolicy (P3) and classify_task.

Coverage:
  * classify_task keyword routing for all task types
  * Construction validation (fallback role must be registered)
  * _dispatch: correct role selected, KeyError fallback, RuntimeError propagation
  * _run_question: happy path and error path
  * compute_routing_summary: role counts, task type counts, per-dataset grouping
"""
from __future__ import annotations

import pytest
from llm_gateway import PromptRequest

from council_policies.models import TaskType
from council_policies.p3_policy import (
    P3QuestionResult,
    RuleBasedRoutingPolicy,
    classify_task,
    compute_routing_summary,
)
from fakes import FakeClient, FakeOrchestrator, make_case, make_orch_response


# ── helpers ────────────────────────────────────────────────────────────────────


def _case(question: str, dataset: str = "test", eid: str = "ex-1"):
    return make_case(question=question, dataset=dataset, eid=eid)


def _orch(*roles: str, texts: dict | None = None, fail: str | None = None) -> FakeOrchestrator:
    texts = texts or {}
    return FakeOrchestrator({
        r: FakeClient(r, text=texts.get(r, f"resp-{r}"), fail=(r == fail))
        for r in roles
    })


# ── classify_task ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prompt,expected", [
    ("solve the equation 2x + 3 = 7", TaskType.MATH),
    ("write a python function to reverse a string code", TaskType.CODE),
    ("analyse and explain why reasoning step-by-step", TaskType.REASONING),
    ("true or false: fact-check this claim", TaskType.FEVER),
    ("what is the capital of France", TaskType.QA),  # no keywords → fallback
    ("", TaskType.QA),                               # empty → fallback
])
def test_classify_task(prompt, expected):
    assert classify_task(prompt) == expected


def test_classify_task_case_insensitive():
    assert classify_task("SOLVE the EQUATION x^2 = 4") == TaskType.MATH


def test_classify_task_picks_highest_scoring_type():
    # "code" keyword appears more than any reasoning keyword
    result = classify_task("implement code for a function code code")
    assert result == TaskType.CODE


# ── Construction ───────────────────────────────────────────────────────────────


def test_construction_raises_when_fallback_not_registered():
    orch = _orch("qa", "reasoning")
    with pytest.raises(ValueError, match="fallback_role"):
        RuleBasedRoutingPolicy(orch, fallback_role="general")


def test_construction_succeeds_with_registered_fallback():
    orch = _orch("general", "qa")
    policy = RuleBasedRoutingPolicy(orch, fallback_role="general")
    assert policy.fallback_role == "general"


def test_construction_default_fallback_is_general():
    orch = _orch("general")
    policy = RuleBasedRoutingPolicy(orch)
    assert policy.fallback_role == "general"


# ── _dispatch ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_routes_to_correct_role():
    # TASK_TO_ROLE[MATH] = "math"; orch has "math" registered
    orch = _orch("math", "general", texts={"math": "math-answer"})
    policy = RuleBasedRoutingPolicy(orch, fallback_role="general")

    req = PromptRequest(user_prompt="solve 2+2 math")
    role, response = await policy._dispatch(TaskType.MATH, req)

    assert role == "math"
    assert response.text == "math-answer"
    assert len(orch.get_client("math").requests) == 1


@pytest.mark.asyncio
async def test_dispatch_falls_back_when_classified_role_not_registered():
    # "reasoning" not in orch; fallback is "general"
    orch = _orch("general", texts={"general": "fallback-answer"})
    policy = RuleBasedRoutingPolicy(orch, fallback_role="general")

    req = PromptRequest(user_prompt="reason about this")
    role, response = await policy._dispatch(TaskType.REASONING, req)

    assert role == "general"
    assert response.text == "fallback-answer"


@pytest.mark.asyncio
async def test_dispatch_raises_runtime_error_when_both_roles_missing():
    # Construct with a valid fallback, then manually drop it so dispatch fails.
    orch = FakeOrchestrator({"general": FakeClient("general")})
    policy = RuleBasedRoutingPolicy(orch, fallback_role="general")
    # Remove the fallback to simulate the "mutated after init" edge case.
    orch._clients.pop("general")

    req = PromptRequest(user_prompt="reason about this")
    with pytest.raises(RuntimeError, match="both missing"):
        await policy._dispatch(TaskType.REASONING, req)


# ── _run_question ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_question_happy_path():
    orch = _orch("code", "general", texts={"code": "sorted-list"})
    policy = RuleBasedRoutingPolicy(orch, fallback_role="general")

    case = _case("implement a code function to sort a list")
    result = await policy._run_question(case)

    assert result is not None
    assert not result.failed
    assert result.task_type == TaskType.CODE
    assert result.routed_role == "code"
    assert result.response is not None
    assert result.response.text == "sorted-list"


@pytest.mark.asyncio
async def test_run_question_returns_error_result_on_client_failure():
    # "math" client raises RuntimeError; result has error set but is not None.
    orch = _orch("math", "general", fail="math")
    policy = RuleBasedRoutingPolicy(orch, fallback_role="general")

    case = _case("solve the equation math x^2 = 4")
    result = await policy._run_question(case)

    assert result is not None
    assert result.failed
    assert result.error is not None


# ── compute_routing_summary ────────────────────────────────────────────────────


def _p3_result(dataset: str, role: str, task_type: TaskType, eid: str = "x") -> P3QuestionResult:
    return P3QuestionResult(
        case=_case("q", dataset=dataset, eid=eid),
        task_type=task_type,
        routed_role=role,
        response=make_orch_response(role, "text"),
    )


def test_compute_routing_summary_role_counts():
    results = [
        _p3_result("ds1", "math", TaskType.MATH, "e1"),
        _p3_result("ds1", "code", TaskType.CODE, "e2"),
        _p3_result("ds1", "math", TaskType.MATH, "e3"),
    ]
    summaries = compute_routing_summary(results)

    assert len(summaries) == 1
    s = summaries[0]
    assert s.dataset_name == "ds1"
    assert s.question_count == 3
    assert s.role_counts["math"] == 2
    assert s.role_counts["code"] == 1
    assert s.task_type_counts[TaskType.MATH.value] == 2
    assert s.task_type_counts[TaskType.CODE.value] == 1


def test_compute_routing_summary_groups_by_dataset():
    results = [
        _p3_result("ds1", "math", TaskType.MATH, "e1"),
        _p3_result("ds2", "code", TaskType.CODE, "e2"),
        _p3_result("ds2", "code", TaskType.CODE, "e3"),
    ]
    summaries = compute_routing_summary(results)

    assert len(summaries) == 2
    by_name = {s.dataset_name: s for s in summaries}
    assert by_name["ds1"].question_count == 1
    assert by_name["ds2"].question_count == 2


def test_compute_routing_summary_sorted_by_name():
    results = [
        _p3_result("zebra", "math", TaskType.MATH, "e1"),
        _p3_result("apple", "code", TaskType.CODE, "e2"),
    ]
    summaries = compute_routing_summary(results)
    assert [s.dataset_name for s in summaries] == ["apple", "zebra"]


def test_compute_routing_summary_avg_latency_for_failed_questions():
    # P3QuestionResult with no response should not contribute to latency.
    failed = P3QuestionResult(
        case=_case("q", dataset="ds1", eid="e1"),
        task_type=TaskType.QA,
        routed_role="general",
        error="boom",
    )
    summaries = compute_routing_summary([failed])
    assert len(summaries) == 1
    assert summaries[0].avg_latency_ms is None
