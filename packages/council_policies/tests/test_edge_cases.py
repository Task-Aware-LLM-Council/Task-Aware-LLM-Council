"""
Edge case tests for council_policies.voter

Covers gaps not addressed in test_voting.py or test_domains.py:
- Model failure during answer gathering (one or all models crash)
- Model failure during voting (one or all models crash)
- Vote parsing with markdown formatting, numbers, gibberish
- All models vote invalid labels → falls to aggregator
- Empty / whitespace-only question raises ValueError
- Aggregator itself fails
- Votes with mixed valid/invalid labels
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from llm_gateway.models import Provider, ProviderConfig

from council_policies.voter import (
    CouncilAnswer,
    CouncilResult,
    ModelConfig,
    Vote,
    _parse_vote,
    gather_answers,
    gather_votes,
    majority_winner,
    run_council,
    tally_votes,
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


def _vote(voted_for: str, voter: str = "m") -> Vote:
    return Vote(voter_model_id=voter, voted_for=voted_for, confidence="High", reason="ok")


def _answer(model: ModelConfig, text: str) -> CouncilAnswer:
    return CouncilAnswer(model_config=model, answer=text)


async def _patched_run_council(
    answer_texts: list[str],
    vote_labels: list[str],
    question: str = "Test?",
    tiebreak_label: str = "A",
) -> CouncilResult:
    answers = [_answer(MODELS[i], answer_texts[i]) for i in range(3)]
    votes = [
        Vote(voter_model_id=MODELS[i].model_id, voted_for=vote_labels[i],
             confidence="High", reason="ok")
        for i in range(3)
    ]
    with patch("council_policies.voter.gather_answers", new=AsyncMock(return_value=answers)), \
         patch("council_policies.voter.gather_votes",   new=AsyncMock(return_value=votes)), \
         patch("council_policies.voter.run_aggregator", new=AsyncMock(return_value=tiebreak_label)):
        return await run_council(MODELS, question)


# ---------------------------------------------------------------------------
# Vote parsing edge cases
# ---------------------------------------------------------------------------

class TestParseVoteEdgeCases:
    def test_vote_with_markdown_bold(self):
        """Model wraps output in markdown: **Vote: B** — regex should still find B."""
        raw = "**Vote: B**\n**Confidence: High**\nReason: Best answer."
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "B"

    def test_vote_with_number_instead_of_label(self):
        """Model says 'Vote: 1' instead of 'Vote: A' — defaults to A."""
        raw = "Vote: 1\nConfidence: High\nReason: First answer is best."
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "A"   # graceful default

    def test_vote_with_word_instead_of_label(self):
        """Model says 'Vote: first' — defaults to A."""
        raw = "Vote: first\nConfidence: Medium\nReason: ok"
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "A"

    def test_completely_unrelated_output(self):
        """Model ignores the format entirely — defaults to A, low confidence."""
        raw = "I think all answers are equally valid and cannot make a decision."
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "A"
        assert v.confidence == "Low"

    def test_vote_label_d_invalid(self):
        """Model votes 'D' which is not a valid label — defaults to A."""
        raw = "Vote: D\nConfidence: High\nReason: none of the above"
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "A"

    def test_vote_appears_multiple_times_first_wins(self):
        """If 'Vote:' appears twice, the first match is used (regex finds first)."""
        raw = "Vote: A\nReason: good\nVote: C\nConfidence: High"
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "A"

    def test_empty_model_output(self):
        """Completely empty response — should not crash, defaults to A."""
        v = _parse_vote("", "m1")
        assert v.voted_for == "A"
        assert v.confidence == "Low"

    def test_only_whitespace_output(self):
        """Only whitespace — should not crash."""
        v = _parse_vote("   \n\t  ", "m1")
        assert v.voted_for == "A"

    def test_vote_c_with_trailing_punctuation(self):
        """Vote: C. — punctuation after label should not break parsing."""
        raw = "Vote: C.\nConfidence: High\nReason: C is correct."
        v = _parse_vote(raw, "m1")
        # regex [ABC] captures C, period is after — should still work
        assert v.voted_for == "C"

    def test_confidence_case_insensitive(self):
        """'confidence: HIGH' → normalized to 'High'."""
        raw = "Vote: B\nconfidence: HIGH\nReason: ok"
        v = _parse_vote(raw, "m1")
        assert v.confidence == "High"

    def test_reason_with_code_block(self):
        """Reason contains a code block with newlines — should be fully captured."""
        raw = (
            "Vote: A\nConfidence: High\nReason: Answer A has correct code:\n"
            "```python\ndef add(a, b): return a + b\n```"
        )
        v = _parse_vote(raw, "m1")
        assert v.voted_for == "A"
        assert "def add" in v.reason


# ---------------------------------------------------------------------------
# Model failure during answer gathering
# ---------------------------------------------------------------------------

class TestAnswerGatheringFailures:
    @pytest.mark.asyncio
    async def test_one_model_fails_during_answer_phase(self, monkeypatch):
        """If one model raises during answer gathering, its answer is empty string.
        The council should still continue with the remaining two answers."""

        call_count = 0

        async def fake_get_answer(model_cfg, question):
            nonlocal call_count
            call_count += 1
            if model_cfg.model_id == "model-b":
                raise RuntimeError("model-b API timeout")
            return CouncilAnswer(model_config=model_cfg, answer=f"Answer from {model_cfg.model_id}")

        with patch("council_policies.voter._get_answer", side_effect=fake_get_answer):
            answers = await gather_answers(MODELS, "What is 2+2?")

        assert len(answers) == 3
        assert answers[0].answer == "Answer from model-a"
        assert answers[1].answer == ""          # failed model → empty string
        assert answers[2].answer == "Answer from model-c"

    @pytest.mark.asyncio
    async def test_all_models_fail_during_answer_phase(self, monkeypatch):
        """All three models fail — all answers are empty strings. Council still resolves."""

        async def fake_get_answer(model_cfg, question):
            raise ConnectionError("network down")

        with patch("council_policies.voter._get_answer", side_effect=fake_get_answer):
            answers = await gather_answers(MODELS, "What is 2+2?")

        assert len(answers) == 3
        assert all(a.answer == "" for a in answers)

    @pytest.mark.asyncio
    async def test_two_models_fail_during_answer_phase(self, monkeypatch):
        """Two of three models fail — only one real answer, two empty."""

        async def fake_get_answer(model_cfg, question):
            if model_cfg.model_id in ("model-a", "model-c"):
                raise TimeoutError("timeout")
            return CouncilAnswer(model_config=model_cfg, answer="42")

        with patch("council_policies.voter._get_answer", side_effect=fake_get_answer):
            answers = await gather_answers(MODELS, "What is 6 * 7?")

        assert answers[0].answer == ""
        assert answers[1].answer == "42"
        assert answers[2].answer == ""


# ---------------------------------------------------------------------------
# Model failure during voting phase
# ---------------------------------------------------------------------------

class TestVotingFailures:
    @pytest.mark.asyncio
    async def test_one_model_fails_during_voting(self, monkeypatch):
        """If one model crashes during voting, it abstains with vote='A', confidence='Low'."""

        async def fake_cast_vote(voter, question, a, b, c):
            if voter.model_id == "model-b":
                raise RuntimeError("model-b rate limited")
            return Vote(voter_model_id=voter.model_id, voted_for="C",
                        confidence="High", reason="C is best")

        answers = [_answer(MODELS[i], f"ans-{i}") for i in range(3)]
        with patch("council_policies.voter._cast_vote", side_effect=fake_cast_vote):
            votes = await gather_votes(MODELS, "Q?", answers)

        assert len(votes) == 3
        assert votes[0].voted_for == "C"
        assert votes[1].voted_for == "A"           # abstain default
        assert votes[1].confidence == "Low"
        assert "Vote failed" in votes[1].reason
        assert votes[2].voted_for == "C"

    @pytest.mark.asyncio
    async def test_all_models_fail_during_voting(self, monkeypatch):
        """All models fail during voting — all abstain to A, triggers aggregator."""

        async def fake_cast_vote(voter, question, a, b, c):
            raise ConnectionError("network down")

        answers = [_answer(MODELS[i], f"ans-{i}") for i in range(3)]
        with patch("council_policies.voter._cast_vote", side_effect=fake_cast_vote):
            votes = await gather_votes(MODELS, "Q?", answers)

        assert all(v.voted_for == "A" for v in votes)
        # All vote A → majority A (3-0), so council resolves without aggregator
        tally = tally_votes(votes)
        assert majority_winner(tally) == "A"

    @pytest.mark.asyncio
    async def test_two_models_fail_voting_one_decides(self, monkeypatch):
        """Two models fail voting → they abstain to A. One real vote for B.
        Result: A=2 (two abstentions), B=1 → A wins by default majority."""

        async def fake_cast_vote(voter, question, a, b, c):
            if voter.model_id in ("model-a", "model-c"):
                raise RuntimeError("failed")
            return Vote(voter_model_id=voter.model_id, voted_for="B",
                        confidence="High", reason="B is correct")

        answers = [_answer(MODELS[i], f"ans-{i}") for i in range(3)]
        with patch("council_policies.voter._cast_vote", side_effect=fake_cast_vote):
            votes = await gather_votes(MODELS, "Q?", answers)

        tally = tally_votes(votes)
        assert tally["A"] == 2   # two abstentions
        assert tally["B"] == 1
        assert majority_winner(tally) == "A"


# ---------------------------------------------------------------------------
# Invalid vote labels in tally
# ---------------------------------------------------------------------------

class TestInvalidVoteLabels:
    def test_all_invalid_votes_produce_zero_tally(self):
        """If all 3 models vote for invalid labels, tally is all zeros → tie → aggregator."""
        votes = [
            _vote("D"), _vote("E"), _vote("F"),
        ]
        tally = tally_votes(votes)
        assert tally == {"A": 0, "B": 0, "C": 0}
        assert majority_winner(tally) is None   # triggers aggregator

    def test_mixed_valid_and_invalid_votes(self):
        """Two models vote invalid, one votes B → B=1, not enough for majority → tie."""
        votes = [_vote("D"), _vote("B"), _vote("Z")]
        tally = tally_votes(votes)
        assert tally["B"] == 1
        assert tally["A"] == 0
        assert majority_winner(tally) is None

    @pytest.mark.asyncio
    async def test_all_invalid_votes_trigger_aggregator(self):
        """End-to-end: all models vote invalid labels → aggregator is used."""
        result = await _patched_run_council(
            answer_texts=["ans-A", "ans-B", "ans-C"],
            vote_labels=["D", "E", "F"],    # all invalid
            tiebreak_label="B",
        )
        assert result.tiebreak_used is True
        assert result.winner_label == "B"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    @pytest.mark.asyncio
    async def test_empty_question_raises(self):
        with pytest.raises(ValueError, match="empty"):
            await run_council(MODELS, "")

    @pytest.mark.asyncio
    async def test_whitespace_only_question_raises(self):
        with pytest.raises(ValueError, match="empty"):
            await run_council(MODELS, "   \n\t  ")

    @pytest.mark.asyncio
    async def test_too_few_models_raises(self):
        with pytest.raises(ValueError, match="exactly 3 models"):
            await run_council(MODELS[:1], "Question?")

    @pytest.mark.asyncio
    async def test_too_many_models_raises(self):
        with pytest.raises(ValueError, match="exactly 3 models"):
            await run_council(MODELS + [_model("extra")], "Question?")

    @pytest.mark.asyncio
    async def test_exactly_three_models_does_not_raise(self):
        """Sanity check — 3 models should not raise the model count error."""
        result = await _patched_run_council(
            answer_texts=["a", "b", "c"],
            vote_labels=["A", "A", "B"],
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Tiebreak / aggregator edge cases
# ---------------------------------------------------------------------------

class TestAggregatorEdgeCases:
    @pytest.mark.asyncio
    async def test_aggregator_returns_none_falls_back_to_first_answer(self):
        """If aggregator returns NONE (all answers wrong), fall back to answers[0]."""
        result = await _patched_run_council(
            answer_texts=["ans-A", "ans-B", "ans-C"],
            vote_labels=["A", "B", "C"],   # tie
            tiebreak_label="NONE",
        )
        assert result.winner_label == "NONE"
        # Code falls back to answers[0] when winner_label is NONE
        assert result.winning_answer == "ans-A"
        assert result.tiebreak_used is True

    @pytest.mark.asyncio
    async def test_tiebreak_not_used_on_majority(self):
        """Aggregator must NOT be called when there is a clear majority."""
        result = await _patched_run_council(
            answer_texts=["ans-A", "ans-B", "ans-C"],
            vote_labels=["B", "B", "A"],
        )
        assert result.tiebreak_used is False
        assert result.winner_label == "B"

    @pytest.mark.asyncio
    async def test_custom_arbitrator_used_on_tie(self):
        """Caller can specify which model acts as arbitrator on a tie."""
        custom_arbitrator = _model("custom-arbitrator")
        answers = [_answer(MODELS[i], f"ans-{i}") for i in range(3)]
        votes = [
            Vote(voter_model_id=MODELS[i].model_id, voted_for=label,
                 confidence="High", reason="ok")
            for i, label in enumerate(["A", "B", "C"])
        ]

        captured_arbitrator = []

        async def fake_aggregator(arbitrator, question, answers, votes):
            captured_arbitrator.append(arbitrator.model_id)
            return "C"

        with patch("council_policies.voter.gather_answers", new=AsyncMock(return_value=answers)), \
             patch("council_policies.voter.gather_votes",   new=AsyncMock(return_value=votes)), \
             patch("council_policies.voter.run_aggregator", side_effect=fake_aggregator):
            result = await run_council(MODELS, "Question?", arbitrator=custom_arbitrator)

        assert captured_arbitrator == ["custom-arbitrator"]
        assert result.winner_label == "C"
        assert result.tiebreak_used is True
