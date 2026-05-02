"""
Tests for `LearnedRouter` — pure decision logic.

The HF-backed `ScoreFn` (distilroberta fine-tune) is NOT exercised
here; it ships alongside the training script. These tests inject a
deterministic callable so the floor/fallback/scoring contract is
pinned without pulling torch into the unit-test run.
"""

from __future__ import annotations

from typing import Callable

import pytest

from council_policies.p4.router import LearnedRouter, RoutingDecision


def _const(scores: dict[str, float]) -> Callable[[str], dict[str, float]]:
    """ScoreFn that returns the same dict regardless of input."""
    return lambda _text: scores


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_rejects_floor_outside_unit_interval():
    with pytest.raises(ValueError, match="floor"):
        LearnedRouter(
            score_fn=_const({"a": 1.0}),
            fallback_role="a",
            floor=1.5,
        )
    with pytest.raises(ValueError, match="floor"):
        LearnedRouter(
            score_fn=_const({"a": 1.0}),
            fallback_role="a",
            floor=-0.1,
        )


def test_rejects_negative_context_cap():
    with pytest.raises(ValueError, match="context_char_cap"):
        LearnedRouter(
            score_fn=_const({"a": 1.0}),
            fallback_role="a",
            context_char_cap=-1,
        )


# --------------------------------------------------------------------------- #
# Happy path — argmax above floor
# --------------------------------------------------------------------------- #


def test_routes_to_argmax_when_above_floor():
    router = LearnedRouter(
        score_fn=_const(
            {"math_code": 0.7, "qa_reasoning": 0.2, "fact_general": 0.1}
        ),
        fallback_role="fact_general",
        floor=0.3,
    )

    decision = router.classify("Compute 2+2")

    assert decision.role == "math_code"
    assert decision.confidence == pytest.approx(0.7)
    assert decision.fallback_used is False
    assert decision.fallback_reason is None
    # Full score distribution preserved — eval needs the shape, not just
    # the winner.
    assert decision.scores == {
        "math_code": 0.7,
        "qa_reasoning": 0.2,
        "fact_general": 0.1,
    }


def test_ties_resolved_by_dict_order():
    """Tie-break is `max()`'s first-seen behavior. Not a correctness
    claim — a regression guard so nobody 'helpfully' changes it without
    noticing the downstream impact on reproducibility."""
    router = LearnedRouter(
        score_fn=_const({"first": 0.5, "second": 0.5}),
        fallback_role="first",
        floor=0.3,
    )

    decision = router.classify("anything")

    assert decision.role == "first"


# --------------------------------------------------------------------------- #
# Below-floor fallback — scoring contract
# --------------------------------------------------------------------------- #


def test_below_floor_routes_to_fallback():
    router = LearnedRouter(
        score_fn=_const(
            {"math_code": 0.25, "qa_reasoning": 0.20, "fact_general": 0.55}
        ),
        fallback_role="fact_general",
        floor=0.7,
    )

    decision = router.classify("anything")

    assert decision.role == "fact_general"
    assert decision.fallback_used is True
    assert decision.fallback_reason == "low_confidence"


def test_below_floor_preserves_learned_scores():
    """Design decision: on fallback, keep the learned distribution
    intact and report `confidence = max(scores.values())`. Rationale:
    the learned scores are the calibration signal for tuning `floor`
    later. Callers distinguish fallback via `fallback_used`, not by
    inspecting `scores`."""
    learned_scores = {
        "math_code": 0.25,
        "qa_reasoning": 0.20,
        "fact_general": 0.10,
    }
    router = LearnedRouter(
        score_fn=_const(learned_scores),
        fallback_role="fact_general",
        floor=0.3,
    )

    decision = router.classify("anything")

    assert decision.scores == learned_scores
    # Top-1 score reported as confidence — not the fallback role's score
    # and not 0.0. "How uncertain was the model" survives the fallback.
    assert decision.confidence == pytest.approx(0.25)


def test_fallback_role_need_not_be_argmax_even_in_score_dict():
    """The fallback role is a policy handle, not an ML label. It may or
    may not appear in the score dict; the router does not enforce it
    one way or the other."""
    router = LearnedRouter(
        score_fn=_const({"math_code": 0.15, "qa_reasoning": 0.12}),
        fallback_role="fact_general",  # absent from scores
        floor=0.3,
    )

    decision = router.classify("anything")

    assert decision.role == "fact_general"
    assert decision.fallback_used is True
    assert "fact_general" not in decision.scores


# --------------------------------------------------------------------------- #
# Boundary conditions
# --------------------------------------------------------------------------- #


def test_score_exactly_at_floor_is_accepted():
    """Strict `<` semantics: `top_score < floor` triggers fallback, so
    `top_score == floor` is accepted. Chosen because a floor of 0.0
    then meaningfully means 'never fall back on confidence' — the
    inclusive variant would make 0.0 unreachable."""
    router = LearnedRouter(
        score_fn=_const({"math_code": 0.30, "qa_reasoning": 0.10}),
        fallback_role="fact_general",
        floor=0.30,
    )

    decision = router.classify("anything")

    assert decision.role == "math_code"
    assert decision.fallback_used is False


def test_floor_zero_never_falls_back_on_confidence():
    """Regression guard on the floor-semantics rationale above."""
    router = LearnedRouter(
        score_fn=_const({"math_code": 0.0001, "qa_reasoning": 0.00005}),
        fallback_role="fact_general",
        floor=0.0,
    )

    decision = router.classify("anything")

    assert decision.role == "math_code"
    assert decision.fallback_used is False


def test_empty_score_dict_falls_back_with_distinct_reason():
    """A broken ScoreFn (empty return) should not crash the policy —
    degrade to fallback with a distinct `fallback_reason` so eval can
    separate model-uncertain from model-broken cases."""
    router = LearnedRouter(
        score_fn=_const({}),
        fallback_role="fact_general",
    )

    decision = router.classify("anything")

    assert decision.role == "fact_general"
    assert decision.fallback_used is True
    assert decision.fallback_reason == "empty_scores"
    assert decision.confidence == 0.0


# --------------------------------------------------------------------------- #
# Input formatting — training/inference parity
# --------------------------------------------------------------------------- #


def test_context_is_appended_when_present():
    """Training-time featurizer must match this format. Any rewrite
    here without retraining silently degrades accuracy."""
    seen: list[str] = []

    def recording_score_fn(text: str) -> dict[str, float]:
        seen.append(text)
        return {"math_code": 0.9}

    router = LearnedRouter(
        score_fn=recording_score_fn,
        fallback_role="fact_general",
    )
    router.classify("Solve x^2 + 3x = 0", context="Assume real roots.")

    assert seen == ["Solve x^2 + 3x = 0\n\nContext:\nAssume real roots."]


def test_context_is_omitted_when_blank():
    seen: list[str] = []

    def recording_score_fn(text: str) -> dict[str, float]:
        seen.append(text)
        return {"math_code": 0.9}

    router = LearnedRouter(
        score_fn=recording_score_fn,
        fallback_role="fact_general",
    )
    router.classify("Solve x^2", context="   ")

    assert seen == ["Solve x^2"]


def test_context_is_truncated_to_char_cap():
    """Upstream guard before tokenization. The tokenizer also truncates;
    this cap just keeps us from paying tokenizer time on text we will
    drop anyway."""
    seen: list[str] = []

    def recording_score_fn(text: str) -> dict[str, float]:
        seen.append(text)
        return {"math_code": 0.9}

    big_context = "x" * 5000
    router = LearnedRouter(
        score_fn=recording_score_fn,
        fallback_role="fact_general",
        context_char_cap=100,
    )
    router.classify("prompt", context=big_context)

    assert len(seen) == 1
    assert seen[0] == "prompt\n\nContext:\n" + "x" * 100


# --------------------------------------------------------------------------- #
# Return shape — RoutingDecision contract parity with KeywordRouter
# --------------------------------------------------------------------------- #


def test_returns_routing_decision_instance():
    """Type parity with `KeywordRouter` — the policy depends on the
    dataclass shape, not the implementation."""
    router = LearnedRouter(
        score_fn=_const({"math_code": 0.9}),
        fallback_role="fact_general",
    )

    decision = router.classify("anything")

    assert isinstance(decision, RoutingDecision)
