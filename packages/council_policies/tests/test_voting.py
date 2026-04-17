"""
Tests for council_policies.voter

Covers core voting mechanics:
- Vote parsing (clean output, messy output, missing fields)
- Majority win (2-1, 3-0)
- Three-way tie → aggregator used
- Tally logic
- run_council() end-to-end with fake clients
- Error handling: model failure during answer gathering, during voting
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from llm_gateway.models import Provider, ProviderConfig

from council_policies.voter import (
    CouncilAnswer,
    ModelConfig,
    Vote,
    majority_winner,
    tally_votes,
    _parse_vote,
    run_council,
)
from fake_client import FakeClient, FailingClient

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _model(model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider_config=ProviderConfig(provider=Provider.OPENROUTER),
    )


MODELS = [_model("model-a"), _model("model-b"), _model("model-c")]


def _vote(voted_for: str, voter: str = "model-x") -> Vote:
    return Vote(voter_model_id=voter, voted_for=voted_for, confidence="High", reason="test")


def _answer(model: ModelConfig, text: str) -> CouncilAnswer:
    return CouncilAnswer(model_config=model, answer=text)


# ---------------------------------------------------------------------------
# Vote parsing
# ---------------------------------------------------------------------------

class TestParseVote:
    def test_parses_clean_a(self):
        raw = "Vote: A\nConfidence: High\nReason: Best answer."
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "A"
        assert v.confidence == "High"
        assert "Best answer" in v.reason

    def test_parses_clean_b(self):
        raw = "Vote: B\nConfidence: Medium\nReason: More complete."
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "B"
        assert v.confidence == "Medium"

    def test_parses_clean_c(self):
        raw = "Vote: C\nConfidence: Low\nReason: Least wrong."
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "C"

    def test_parses_lowercase_vote(self):
        raw = "vote: b\nconfidence: high\nreason: solid."
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "B"

    def test_missing_vote_field_defaults_to_a(self):
        """Graceful degradation when model ignores format."""
        raw = "I think the best answer is probably B but I'm not sure."
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "A"   # default

    def test_missing_confidence_defaults_to_low(self):
        raw = "Vote: C\nReason: Only correct one."
        v = _parse_vote(raw, "m1")
        assert v.confidence == "Low"

    def test_multiline_reason_captured(self):
        raw = "Vote: A\nConfidence: High\nReason: Answer A is correct because:\n- Step 1\n- Step 2"
        v = _parse_vote(raw, "m1")
        assert "Step 1" in v.reason
        assert "Step 2" in v.reason

    def test_voter_model_id_stored(self):
        raw = "Vote: B\nConfidence: Low\nReason: ok"
        v = _parse_vote(raw, "gpt-4o")
        assert v.voter_model_id == "gpt-4o"


# ---------------------------------------------------------------------------
# Tally and majority
# ---------------------------------------------------------------------------

class TestTallyAndMajority:
    def test_unanimous_vote(self):
        votes = [_vote("A"), _vote("A"), _vote("A")]
        tally = tally_votes(votes)
        assert tally == {"A": 3, "B": 0, "C": 0}
        assert majority_winner(tally) == "A"

    def test_majority_2_1_a_wins(self):
        votes = [_vote("A"), _vote("A"), _vote("B")]
        assert majority_winner(tally_votes(votes)) == "A"

    def test_majority_2_1_b_wins(self):
        votes = [_vote("B"), _vote("A"), _vote("B")]
        assert majority_winner(tally_votes(votes)) == "B"

    def test_majority_2_1_c_wins(self):
        votes = [_vote("C"), _vote("C"), _vote("A")]
        assert majority_winner(tally_votes(votes)) == "C"

    def test_three_way_tie_returns_none(self):
        votes = [_vote("A"), _vote("B"), _vote("C")]
        assert majority_winner(tally_votes(votes)) is None

    def test_tally_counts_all_labels(self):
        votes = [_vote("B"), _vote("B"), _vote("C")]
        tally = tally_votes(votes)
        assert tally["A"] == 0
        assert tally["B"] == 2
        assert tally["C"] == 1


# ---------------------------------------------------------------------------
# run_council() end-to-end (mocked clients)
# ---------------------------------------------------------------------------

def _make_answer_response(text: str) -> str:
    return text


def _make_vote_response(label: str) -> str:
    return f"Vote: {label}\nConfidence: High\nReason: Best answer."


async def _patched_run_council(
    answer_texts: list[str],
    vote_labels: list[str],
    question: str = "Test question?",
    tiebreak_label: str | None = None,
) -> object:
    """
    Helper that patches gather_answers, gather_votes, and run_aggregator
    so we can test run_council() logic without real HTTP calls.
    """
    answers = [_answer(MODELS[i], answer_texts[i]) for i in range(3)]
    votes = [
        Vote(voter_model_id=MODELS[i].model_id, voted_for=vote_labels[i],
             confidence="High", reason="ok")
        for i in range(3)
    ]

    with patch("council_policies.voter.gather_answers", new=AsyncMock(return_value=answers)), \
         patch("council_policies.voter.gather_votes", new=AsyncMock(return_value=votes)), \
         patch("council_policies.voter.run_aggregator",
               new=AsyncMock(return_value=tiebreak_label or "A")):
        return await run_council(MODELS, question)


class TestRunCouncil:
    @pytest.mark.asyncio
    async def test_majority_2_1_picks_correct_winner(self):
        result = await _patched_run_council(
            answer_texts=["ans-A", "ans-B", "ans-C"],
            vote_labels=["A", "A", "B"],
        )
        assert result.winner_label == "A"
        assert result.winning_answer == "ans-A"
        assert result.winning_model_id == MODELS[0].model_id
        assert result.tiebreak_used is False

    @pytest.mark.asyncio
    async def test_unanimous_vote(self):
        result = await _patched_run_council(
            answer_texts=["ans-A", "ans-B", "ans-C"],
            vote_labels=["C", "C", "C"],
        )
        assert result.winner_label == "C"
        assert result.winning_answer == "ans-C"
        assert result.tiebreak_used is False

    @pytest.mark.asyncio
    async def test_tie_triggers_aggregator(self):
        result = await _patched_run_council(
            answer_texts=["ans-A", "ans-B", "ans-C"],
            vote_labels=["A", "B", "C"],
            tiebreak_label="B",
        )
        assert result.winner_label == "B"
        assert result.winning_answer == "ans-B"
        assert result.tiebreak_used is True

    @pytest.mark.asyncio
    async def test_result_contains_all_answers(self):
        result = await _patched_run_council(
            answer_texts=["ans-A", "ans-B", "ans-C"],
            vote_labels=["A", "A", "B"],
        )
        assert len(result.answers) == 3
        assert result.answers[0].answer == "ans-A"
        assert result.answers[1].answer == "ans-B"
        assert result.answers[2].answer == "ans-C"

    @pytest.mark.asyncio
    async def test_result_contains_all_votes(self):
        result = await _patched_run_council(
            answer_texts=["ans-A", "ans-B", "ans-C"],
            vote_labels=["A", "A", "B"],
        )
        assert len(result.votes) == 3

    @pytest.mark.asyncio
    async def test_wrong_number_of_models_raises(self):
        with pytest.raises(ValueError, match="exactly 3 models"):
            await run_council(MODELS[:2], "question")

    @pytest.mark.asyncio
    async def test_question_stored_in_result(self):
        result = await _patched_run_council(
            answer_texts=["a", "b", "c"],
            vote_labels=["B", "B", "A"],
            question="What is the speed of light?",
        )
        assert result.question == "What is the speed of light?"
