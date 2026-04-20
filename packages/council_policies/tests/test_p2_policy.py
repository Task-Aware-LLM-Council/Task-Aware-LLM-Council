"""
Tests for DatasetCouncilPolicy (P2) and supporting utilities.

Coverage:
  * Construction validation (exactly 3 distinct roles, all registered)
  * _gather_answers: fan-out to 3 models, partial-failure tolerance
  * _gather_ratings: one rating result per role, bad JSON → failed result
  * P2QuestionResult.best_answer: picks role with highest average score
  * compute_dataset_votes: per-dataset winner and aggregation
"""
from __future__ import annotations

import pytest

from council_policies.p2_policy import (
    DatasetCouncilPolicy,
    ModelAnswer,
    P2QuestionResult,
    RatingEntry,
    RatingResult,
    compute_dataset_votes,
)
from fakes import FakeClient, FakeOrchestrator, make_case, make_orch_response


# ── helpers ────────────────────────────────────────────────────────────────────


def _case(question: str = "What is 2+2?", dataset: str = "test", eid: str = "ex-1"):
    return make_case(question=question, dataset=dataset, eid=eid)


_VALID_RATING_JSON = (
    '{"A": {"score": 8, "reasoning": "good"},'
    ' "B": {"score": 6, "reasoning": "ok"},'
    ' "C": {"score": 7, "reasoning": "fine"}}'
)

_THREE_ROLES = ("qa", "reasoning", "general")


def _three_orch(texts: dict | None = None, fail: str | None = None) -> FakeOrchestrator:
    texts = texts or {}
    return FakeOrchestrator({
        r: FakeClient(r, text=texts.get(r, f"answer-{r}"), fail=(r == fail))
        for r in _THREE_ROLES
    })


def _rating_orch(rating_text: str = _VALID_RATING_JSON) -> FakeOrchestrator:
    return FakeOrchestrator({
        r: FakeClient(r, text=rating_text) for r in _THREE_ROLES
    })


# ── Construction ───────────────────────────────────────────────────────────────


def test_construction_requires_exactly_3_roles():
    orch = FakeOrchestrator({"qa": FakeClient("qa"), "reasoning": FakeClient("reasoning")})
    with pytest.raises(ValueError, match="3"):
        DatasetCouncilPolicy(orch, council_roles=("qa", "reasoning"))


def test_construction_requires_distinct_roles():
    orch = _three_orch()
    with pytest.raises(ValueError, match="distinct"):
        DatasetCouncilPolicy(orch, council_roles=("qa", "qa", "general"))


def test_construction_validates_all_roles_registered():
    orch = FakeOrchestrator({
        "qa": FakeClient("qa"),
        "reasoning": FakeClient("reasoning"),
        # "ghost" is NOT registered
    })
    with pytest.raises(ValueError, match="not registered"):
        DatasetCouncilPolicy(orch, council_roles=("qa", "reasoning", "ghost"))


def test_construction_succeeds_with_valid_config():
    orch = _three_orch()
    policy = DatasetCouncilPolicy(orch)
    assert set(policy.council_roles) == {"qa", "reasoning", "general"}


# ── _gather_answers ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gather_answers_fans_out_to_all_three_roles():
    orch = _three_orch(texts={"qa": "QA-ans", "reasoning": "R-ans", "general": "G-ans"})
    policy = DatasetCouncilPolicy(orch)
    answers = await policy._gather_answers(_case())

    assert len(answers) == 3
    assert {a.role for a in answers} == {"qa", "reasoning", "general"}
    assert all(not a.failed for a in answers)


@pytest.mark.asyncio
async def test_gather_answers_marks_failed_client():
    orch = _three_orch(fail="qa")
    policy = DatasetCouncilPolicy(orch)
    answers = await policy._gather_answers(_case())

    failed = [a for a in answers if a.failed]
    assert len(failed) == 1
    assert failed[0].role == "qa"
    assert failed[0].error is not None


@pytest.mark.asyncio
async def test_gather_answers_empty_response_is_failure():
    orch = FakeOrchestrator({
        "qa": FakeClient("qa", text="   "),       # whitespace-only → failed
        "reasoning": FakeClient("reasoning", text="good"),
        "general": FakeClient("general", text="good"),
    })
    policy = DatasetCouncilPolicy(orch)
    answers = await policy._gather_answers(_case())

    qa_ans = next(a for a in answers if a.role == "qa")
    assert qa_ans.failed


# ── _gather_ratings ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gather_ratings_returns_one_result_per_rater():
    orch = _rating_orch(_VALID_RATING_JSON)
    policy = DatasetCouncilPolicy(orch)

    answers = [ModelAnswer(role=r, text=f"ans-{r}") for r in _THREE_ROLES]
    ratings = await policy._gather_ratings(_case(), answers)

    assert len(ratings) == 3
    assert {r.rater_role for r in ratings} == {"qa", "reasoning", "general"}
    assert all(not r.failed for r in ratings)
    assert all(len(r.ratings) == 3 for r in ratings)


@pytest.mark.asyncio
async def test_gather_ratings_marks_bad_json_as_failed():
    orch = _rating_orch("not valid json at all")
    policy = DatasetCouncilPolicy(orch)

    answers = [ModelAnswer(role=r, text="ans") for r in _THREE_ROLES]
    ratings = await policy._gather_ratings(_case(), answers)

    assert all(r.failed for r in ratings)
    assert all(r.error == "could not parse rating JSON" for r in ratings)


# ── P2QuestionResult.best_answer ───────────────────────────────────────────────


def test_best_answer_picks_highest_average_score():
    case = _case()
    answers = [ModelAnswer(role=r, text=f"ans-{r}") for r in _THREE_ROLES]

    # rater 1: qa=9, reasoning=5, general=6
    r1 = RatingResult(
        rater_role="qa",
        label_to_role={"A": "qa", "B": "reasoning", "C": "general"},
        ratings=[RatingEntry("A", 9.0), RatingEntry("B", 5.0), RatingEntry("C", 6.0)],
    )
    # rater 2: qa=8, reasoning=4, general=7
    r2 = RatingResult(
        rater_role="reasoning",
        label_to_role={"A": "qa", "B": "reasoning", "C": "general"},
        ratings=[RatingEntry("A", 8.0), RatingEntry("B", 4.0), RatingEntry("C", 7.0)],
    )
    # qa avg = (9+8)/2 = 8.5 → winner
    result = P2QuestionResult(case=case, answers=answers, ratings=[r1, r2])
    best = result.best_answer

    assert best is not None
    assert best.role == "qa"


def test_best_answer_returns_none_when_all_raters_failed():
    case = _case()
    answers = [ModelAnswer(role="qa", text="ans")]
    ratings = [RatingResult(rater_role="qa", label_to_role={}, ratings=[], error="boom")]

    result = P2QuestionResult(case=case, answers=answers, ratings=ratings)
    assert result.best_answer is None


def test_best_answer_skips_failed_raters():
    case = _case()
    answers = [ModelAnswer(role=r, text=f"ans-{r}") for r in ["qa", "reasoning"]]

    good = RatingResult(
        rater_role="qa",
        label_to_role={"A": "qa", "B": "reasoning"},
        ratings=[RatingEntry("A", 9.0), RatingEntry("B", 3.0)],
    )
    bad = RatingResult(rater_role="reasoning", label_to_role={}, ratings=[], error="failed")

    result = P2QuestionResult(case=case, answers=answers, ratings=[good, bad])
    best = result.best_answer

    assert best is not None
    assert best.role == "qa"


# ── compute_dataset_votes ──────────────────────────────────────────────────────


def _qr_with_rating(dataset: str, label_to_role: dict, scores: dict, eid: str = "x"):
    case = _case(dataset=dataset, eid=eid)
    answers = [ModelAnswer(role=r, text="t") for r in label_to_role.values()]
    entries = [RatingEntry(lbl, sc) for lbl, sc in scores.items()]
    rating = RatingResult(rater_role="qa", label_to_role=label_to_role, ratings=entries)
    return P2QuestionResult(case=case, answers=answers, ratings=[rating])


def test_compute_dataset_votes_picks_winner():
    # Both questions: qa=9, reasoning=5 → qa wins overall avg=(9+9)/2=9
    qr1 = _qr_with_rating("ds1", {"A": "qa", "B": "reasoning"}, {"A": 9.0, "B": 5.0}, "e1")
    qr2 = _qr_with_rating("ds1", {"A": "qa", "B": "reasoning"}, {"A": 9.0, "B": 5.0}, "e2")

    votes = compute_dataset_votes([qr1, qr2])

    assert len(votes) == 1
    v = votes[0]
    assert v.winner == "qa"
    assert v.question_count == 2
    assert abs(v.scores_by_role["qa"] - 9.0) < 0.01


def test_compute_dataset_votes_groups_by_dataset():
    qr1 = _qr_with_rating("ds1", {"A": "qa"}, {"A": 8.0}, "e1")
    qr2 = _qr_with_rating("ds2", {"A": "reasoning"}, {"A": 7.0}, "e2")

    votes = compute_dataset_votes([qr1, qr2])

    assert len(votes) == 2
    by_name = {v.dataset_name: v for v in votes}
    assert by_name["ds1"].winner == "qa"
    assert by_name["ds2"].winner == "reasoning"


def test_compute_dataset_votes_sorted_by_dataset_name():
    qr1 = _qr_with_rating("zebra", {"A": "qa"}, {"A": 5.0}, "e1")
    qr2 = _qr_with_rating("apple", {"A": "qa"}, {"A": 5.0}, "e2")

    votes = compute_dataset_votes([qr1, qr2])
    assert [v.dataset_name for v in votes] == ["apple", "zebra"]


def test_compute_dataset_votes_skips_failed_raters():
    case = _case(dataset="ds1", eid="e1")
    answers = [ModelAnswer(role="qa", text="ans")]
    failed_rating = RatingResult(rater_role="qa", label_to_role={}, ratings=[], error="boom")
    qr = P2QuestionResult(case=case, answers=answers, ratings=[failed_rating])

    votes = compute_dataset_votes([qr])
    # No valid ratings → no entries in totals → empty summaries
    assert votes == []
