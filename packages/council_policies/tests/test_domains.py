"""
Domain-specific council voting tests.

Each domain (MATH, CODING, LOGIC, ENGLISH) has:
  - A realistic question with one clearly correct and two wrong/weaker answers
  - Tests asserting the council picks the correct answer via majority vote
  - Per-domain model performance tracking (which model voted correctly)

These tests use mocked clients so no real API calls are made.
The "domain tracker" fixture accumulates vote data across tests so you can
inspect which fake model consistently voted for correct answers per domain.
"""

from __future__ import annotations

import pytest
from collections import defaultdict
from unittest.mock import AsyncMock, patch

from llm_gateway.models import Provider, ProviderConfig

from council_policies.voter import (
    CouncilAnswer,
    ModelConfig,
    Vote,
    CouncilResult,
    run_council,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(model_id: str) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider_config=ProviderConfig(provider=Provider.OPENROUTER),
    )


# Simulated council: three different "specialist" models
MODEL_MATH    = _model("math-specialist")
MODEL_CODE    = _model("code-specialist")
MODEL_GENERAL = _model("general-model")
MODELS = [MODEL_MATH, MODEL_CODE, MODEL_GENERAL]


async def _run_with_fixed_answers_and_votes(
    question: str,
    answer_a: str,
    answer_b: str,
    answer_c: str,
    vote_a: str,   # what MODEL_MATH votes
    vote_b: str,   # what MODEL_CODE votes
    vote_c: str,   # what MODEL_GENERAL votes
    tiebreak: str = "A",
) -> CouncilResult:
    answers = [
        CouncilAnswer(model_config=MODEL_MATH,    answer=answer_a),
        CouncilAnswer(model_config=MODEL_CODE,    answer=answer_b),
        CouncilAnswer(model_config=MODEL_GENERAL, answer=answer_c),
    ]
    votes = [
        Vote(voter_model_id=MODEL_MATH.model_id,    voted_for=vote_a, confidence="High", reason=""),
        Vote(voter_model_id=MODEL_CODE.model_id,    voted_for=vote_b, confidence="High", reason=""),
        Vote(voter_model_id=MODEL_GENERAL.model_id, voted_for=vote_c, confidence="High", reason=""),
    ]
    with patch("council_policies.voter.gather_answers", new=AsyncMock(return_value=answers)), \
         patch("council_policies.voter.gather_votes",   new=AsyncMock(return_value=votes)), \
         patch("council_policies.voter.run_aggregator", new=AsyncMock(return_value=tiebreak)):
        return await run_council(MODELS, question)


# ---------------------------------------------------------------------------
# Domain tracker fixture
# ---------------------------------------------------------------------------

# Global accumulator: domain → model_id → correct_votes count
_domain_scores: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))


def _record_votes(domain: str, result: CouncilResult, correct_label: str) -> None:
    """Record which models voted for the correct answer."""
    for vote in result.votes:
        if vote.voted_for == correct_label:
            _domain_scores[domain][vote.voter_model_id] += 1


def pytest_sessionfinish(session, exitstatus):
    """Print per-domain model accuracy summary at end of test session."""
    if not _domain_scores:
        return
    print("\n\n=== Council Domain Performance (which model voted correctly) ===")
    for domain, scores in sorted(_domain_scores.items()):
        print(f"\n  [{domain.upper()}]")
        for model_id, count in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"    {model_id}: {count} correct vote(s)")


# ---------------------------------------------------------------------------
# MATH domain
# ---------------------------------------------------------------------------

class TestMathDomain:
    @pytest.mark.asyncio
    async def test_simple_arithmetic(self):
        """2 + 2 = 4, not 5 or 22."""
        result = await _run_with_fixed_answers_and_votes(
            question="What is 2 + 2?",
            answer_a="4",           # correct (MODEL_MATH)
            answer_b="5",           # wrong
            answer_c="22",          # wrong (concatenation mistake)
            vote_a="A", vote_b="A", vote_c="B",  # A wins 2-1
        )
        _record_votes("math", result, correct_label="A")
        assert result.winner_label == "A"
        assert result.winning_answer == "4"
        assert result.tiebreak_used is False

    @pytest.mark.asyncio
    async def test_derivative(self):
        """d/dx of x^2 = 2x."""
        result = await _run_with_fixed_answers_and_votes(
            question="What is the derivative of x^2?",
            answer_a="2x",          # correct
            answer_b="x^2",         # wrong (forgot to differentiate)
            answer_c="2x^2",        # wrong (wrong coefficient)
            vote_a="A", vote_b="A", vote_c="C",
        )
        _record_votes("math", result, correct_label="A")
        assert result.winner_label == "A"
        assert result.winning_answer == "2x"

    @pytest.mark.asyncio
    async def test_all_wrong_uses_aggregator(self):
        """When all models give wrong math answers, 3-way tie → aggregator picks best."""
        result = await _run_with_fixed_answers_and_votes(
            question="What is sqrt(144)?",
            answer_a="11",          # wrong
            answer_b="13",          # wrong
            answer_c="14",          # wrong
            vote_a="A", vote_b="B", vote_c="C",   # 1-1-1 tie
            tiebreak="C",
        )
        _record_votes("math", result, correct_label="NONE")  # no correct answer
        assert result.tiebreak_used is True
        assert result.winner_label == "C"

    @pytest.mark.asyncio
    async def test_multi_step_math(self):
        """(5 * 3) - 7 = 8."""
        result = await _run_with_fixed_answers_and_votes(
            question="What is (5 * 3) - 7?",
            answer_a="8",           # correct
            answer_b="7",           # off by one
            answer_c="8",           # also correct (duplicate correct answer)
            vote_a="A", vote_b="C", vote_c="A",
        )
        _record_votes("math", result, correct_label="A")
        assert result.winner_label == "A"
        assert result.winning_answer == "8"

    @pytest.mark.asyncio
    async def test_fraction_answer(self):
        """1/3 + 1/6 = 1/2."""
        result = await _run_with_fixed_answers_and_votes(
            question="What is 1/3 + 1/6?",
            answer_a="1/2",         # correct
            answer_b="2/9",         # wrong
            answer_c="1/2",         # also correct
            vote_a="A", vote_b="A", vote_c="C",
        )
        _record_votes("math", result, correct_label="A")
        assert result.winner_label == "A"


# ---------------------------------------------------------------------------
# CODING domain
# ---------------------------------------------------------------------------

class TestCodingDomain:
    @pytest.mark.asyncio
    async def test_reverse_string(self):
        """Best Python string reverse is s[::-1]."""
        result = await _run_with_fixed_answers_and_votes(
            question="Write a Python one-liner to reverse a string s.",
            answer_a="s[::-1]",                    # correct & idiomatic
            answer_b="''.join(reversed(s))",       # correct but verbose
            answer_c="s.reverse()",                # wrong: str has no .reverse()
            vote_a="A", vote_b="A", vote_c="C",
        )
        _record_votes("coding", result, correct_label="A")
        assert result.winner_label == "A"
        assert "[::-1]" in result.winning_answer

    @pytest.mark.asyncio
    async def test_fibonacci_function(self):
        """Correct Fibonacci handles base cases."""
        correct_fib = "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)"
        wrong_fib_1 = "def fib(n):\n    return fib(n-1) + fib(n-2)"  # missing base case → infinite recursion
        wrong_fib_2 = "def fib(n):\n    return n * (n-1)"            # wrong formula

        result = await _run_with_fixed_answers_and_votes(
            question="Write a recursive Python function to compute the nth Fibonacci number.",
            answer_a=correct_fib,
            answer_b=wrong_fib_1,
            answer_c=wrong_fib_2,
            vote_a="A", vote_b="A", vote_c="B",
        )
        _record_votes("coding", result, correct_label="A")
        assert result.winner_label == "A"
        assert "if n <= 1" in result.winning_answer

    @pytest.mark.asyncio
    async def test_sql_query(self):
        """SELECT with WHERE clause."""
        result = await _run_with_fixed_answers_and_votes(
            question="Write a SQL query to select all users older than 30.",
            answer_a="SELECT * FROM users WHERE age > 30;",   # correct
            answer_b="SELECT * FROM users WHERE age < 30;",   # wrong operator
            answer_c="SELECT age FROM users WHERE age > 30;", # partial (not all cols)
            vote_a="A", vote_b="A", vote_c="C",
        )
        _record_votes("coding", result, correct_label="A")
        assert result.winner_label == "A"
        assert "age > 30" in result.winning_answer

    @pytest.mark.asyncio
    async def test_off_by_one_caught_by_vote(self):
        """Off-by-one in loop — majority catches it."""
        result = await _run_with_fixed_answers_and_votes(
            question="Print numbers 1 to 5 inclusive in Python.",
            answer_a="for i in range(1, 6): print(i)",   # correct
            answer_b="for i in range(1, 5): print(i)",   # off-by-one (misses 5)
            answer_c="for i in range(0, 5): print(i)",   # off-by-one (starts at 0)
            vote_a="A", vote_b="A", vote_c="B",
        )
        _record_votes("coding", result, correct_label="A")
        assert result.winner_label == "A"

    @pytest.mark.asyncio
    async def test_all_models_give_same_code(self):
        """If all produce same correct answer, winner is still A (first)."""
        code = "lambda x: x * 2"
        result = await _run_with_fixed_answers_and_votes(
            question="Write a Python lambda that doubles its input.",
            answer_a=code, answer_b=code, answer_c=code,
            vote_a="A", vote_b="A", vote_c="A",
        )
        _record_votes("coding", result, correct_label="A")
        assert result.winner_label == "A"
        assert result.winning_answer == code


# ---------------------------------------------------------------------------
# LOGICAL REASONING domain
# ---------------------------------------------------------------------------

class TestLogicDomain:
    @pytest.mark.asyncio
    async def test_syllogism(self):
        """Classic syllogism: Socrates is mortal."""
        result = await _run_with_fixed_answers_and_votes(
            question="All men are mortal. Socrates is a man. Is Socrates mortal?",
            answer_a="Yes, Socrates is mortal.",             # correct
            answer_b="We cannot know without more data.",    # wrong (ignores premises)
            answer_c="No, Socrates is a philosopher.",       # wrong (irrelevant)
            vote_a="A", vote_b="A", vote_c="B",
        )
        _record_votes("logic", result, correct_label="A")
        assert result.winner_label == "A"

    @pytest.mark.asyncio
    async def test_conditional_logic(self):
        """If P then Q. P is true. Therefore Q."""
        result = await _run_with_fixed_answers_and_votes(
            question="If it rains, the ground gets wet. It is raining. What can we conclude?",
            answer_a="The ground is wet.",                    # correct (modus ponens)
            answer_b="It might get wet.",                     # hedged/wrong
            answer_c="We need more information.",             # wrong
            vote_a="A", vote_b="A", vote_c="B",
        )
        _record_votes("logic", result, correct_label="A")
        assert result.winner_label == "A"

    @pytest.mark.asyncio
    async def test_contradiction_detection(self):
        """Contradictory premises should be flagged."""
        result = await _run_with_fixed_answers_and_votes(
            question="A is greater than B. B is greater than A. What is true?",
            answer_a="This is a contradiction — both cannot be true simultaneously.",
            answer_b="A is greater than B.",
            answer_c="B is greater than A.",
            vote_a="A", vote_b="A", vote_c="C",
        )
        _record_votes("logic", result, correct_label="A")
        assert result.winner_label == "A"
        assert "contradiction" in result.winning_answer.lower()

    @pytest.mark.asyncio
    async def test_deductive_chain(self):
        """Multi-step deduction."""
        result = await _run_with_fixed_answers_and_votes(
            question=(
                "All birds have wings. Penguins are birds. "
                "Do penguins have wings?"
            ),
            answer_a="Yes, penguins have wings (though they cannot fly).",   # correct
            answer_b="No, penguins cannot fly so they have no wings.",        # wrong
            answer_c="Only flying birds have wings.",                         # wrong
            vote_a="A", vote_b="A", vote_c="B",
        )
        _record_votes("logic", result, correct_label="A")
        assert result.winner_label == "A"


# ---------------------------------------------------------------------------
# ENGLISH / WRITING domain
# ---------------------------------------------------------------------------

class TestEnglishDomain:
    @pytest.mark.asyncio
    async def test_passive_voice_conversion(self):
        """Active → passive voice transformation."""
        result = await _run_with_fixed_answers_and_votes(
            question="Rewrite in passive voice: 'The cat chased the mouse.'",
            answer_a="The mouse was chased by the cat.",    # correct
            answer_b="The cat was chased by the mouse.",    # reversed subject/object
            answer_c="The mouse chased the cat.",           # just reversed, not passive
            vote_a="A", vote_b="A", vote_c="C",
        )
        _record_votes("english", result, correct_label="A")
        assert result.winner_label == "A"
        assert "was chased by the cat" in result.winning_answer

    @pytest.mark.asyncio
    async def test_grammar_correction(self):
        """Fix subject-verb agreement."""
        result = await _run_with_fixed_answers_and_votes(
            question="Fix the grammar: 'She don't know the answer.'",
            answer_a="She doesn't know the answer.",   # correct
            answer_b="She do not know the answer.",    # still wrong
            answer_c="She didn't knew the answer.",    # wrong tense
            vote_a="A", vote_b="A", vote_c="B",
        )
        _record_votes("english", result, correct_label="A")
        assert result.winner_label == "A"
        assert "doesn't" in result.winning_answer

    @pytest.mark.asyncio
    async def test_synonym_choice(self):
        """Best synonym for 'happy' in a formal context."""
        result = await _run_with_fixed_answers_and_votes(
            question="Give the most formal synonym for 'happy'.",
            answer_a="Elated",      # strong but formal
            answer_b="Glad",        # informal
            answer_c="Content",     # formal and precise
            vote_a="C", vote_b="C", vote_c="A",   # C wins 2-1
        )
        _record_votes("english", result, correct_label="C")
        assert result.winner_label == "C"
        assert result.winning_answer == "Content"

    @pytest.mark.asyncio
    async def test_summary_quality(self):
        """Concise and accurate summary beats verbose/inaccurate ones."""
        result = await _run_with_fixed_answers_and_votes(
            question="Summarize: 'The sun is a star at the center of our solar system.'",
            answer_a="The sun is a central star in our solar system.",   # concise and accurate
            answer_b="The sun is a planet.",                             # factually wrong
            answer_c="There is a sun and it is in space somewhere.",     # vague
            vote_a="A", vote_b="A", vote_c="C",
        )
        _record_votes("english", result, correct_label="A")
        assert result.winner_label == "A"


# ---------------------------------------------------------------------------
# Edge cases across all domains
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_answers_still_resolve(self):
        """Council must not crash when one or more answers are empty strings."""
        result = await _run_with_fixed_answers_and_votes(
            question="What is 1 + 1?",
            answer_a="2",
            answer_b="",           # model returned nothing
            answer_c="",           # model returned nothing
            vote_a="A", vote_b="A", vote_c="A",
        )
        assert result.winner_label == "A"
        assert result.winning_answer == "2"

    @pytest.mark.asyncio
    async def test_very_long_answers_handled(self):
        """Long verbose answers should not break the voting pipeline."""
        long_answer = "The answer is 42. " * 500
        result = await _run_with_fixed_answers_and_votes(
            question="What is the answer to life, the universe, and everything?",
            answer_a=long_answer,
            answer_b="42",
            answer_c="Unknown",
            vote_a="B", vote_b="B", vote_c="A",
        )
        assert result.winner_label == "B"
        assert result.winning_answer == "42"

    @pytest.mark.asyncio
    async def test_identical_answers_from_all_models(self):
        """If all models give same answer, A wins unanimously."""
        result = await _run_with_fixed_answers_and_votes(
            question="What color is the sky?",
            answer_a="Blue", answer_b="Blue", answer_c="Blue",
            vote_a="A", vote_b="A", vote_c="A",
        )
        assert result.winner_label == "A"
        assert result.winning_answer == "Blue"
        assert result.tiebreak_used is False

    @pytest.mark.asyncio
    async def test_special_characters_in_answers(self):
        """Answers with special chars (math symbols, code) must pass through unchanged."""
        result = await _run_with_fixed_answers_and_votes(
            question="Write the quadratic formula.",
            answer_a="x = (-b ± √(b²-4ac)) / 2a",
            answer_b="x = -b/2a",
            answer_c="x = b² - 4ac",
            vote_a="A", vote_b="A", vote_c="B",
        )
        assert result.winner_label == "A"
        assert "±" in result.winning_answer

    @pytest.mark.asyncio
    async def test_multilingual_answer_handled(self):
        """Non-ASCII content should not crash the pipeline."""
        result = await _run_with_fixed_answers_and_votes(
            question="How do you say 'hello' in Japanese?",
            answer_a="こんにちは (Konnichiwa)",   # correct
            answer_b="Bonjour",                    # wrong language
            answer_c="Hola",                       # wrong language
            vote_a="A", vote_b="A", vote_c="B",
        )
        assert result.winner_label == "A"
        assert "こんにちは" in result.winning_answer

    @pytest.mark.asyncio
    @pytest.mark.parametrize("domain,question,correct,wrong1,wrong2", [
        (
            "math",
            "What is 10 % 3?",
            "1",
            "3",
            "0",
        ),
        (
            "coding",
            "What keyword exits a loop in Python?",
            "break",
            "exit",
            "stop",
        ),
        (
            "logic",
            "If A implies B and B implies C, does A imply C?",
            "Yes, by transitivity.",
            "No, A and C are unrelated.",
            "Only if A equals C.",
        ),
        (
            "english",
            "What is the plural of 'ox'?",
            "oxen",
            "oxes",
            "ox's",
        ),
    ])
    async def test_council_picks_correct_per_domain(
        self, domain, question, correct, wrong1, wrong2
    ):
        """Parametrized: council picks the correct answer across all four domains."""
        result = await _run_with_fixed_answers_and_votes(
            question=question,
            answer_a=correct,
            answer_b=wrong1,
            answer_c=wrong2,
            vote_a="A", vote_b="A", vote_c="B",
        )
        _record_votes(domain, result, correct_label="A")
        assert result.winner_label == "A"
        assert result.winning_answer == correct
