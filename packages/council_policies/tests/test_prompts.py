"""
Tests for council_policies.prompts

Covers:
- Correct variable substitution in voter prompt
- Correct variable substitution in aggregator prompt
- Anonymization: answer labels are A/B/C, never model names
- All domain types appear correctly in prompt output
"""

from __future__ import annotations

import pytest

from council_policies.prompts import build_aggregator_prompt, build_voter_prompt

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Voter prompt
# ---------------------------------------------------------------------------

def test_voter_prompt_contains_question():
    prompt = build_voter_prompt(
        question="What is 2 + 2?",
        answer_a="4",
        answer_b="5",
        answer_c="3",
    )
    assert "What is 2 + 2?" in prompt


def test_voter_prompt_contains_all_answers():
    prompt = build_voter_prompt(
        question="Q",
        answer_a="Answer from model one",
        answer_b="Answer from model two",
        answer_c="Answer from model three",
    )
    assert "Answer from model one" in prompt
    assert "Answer from model two" in prompt
    assert "Answer from model three" in prompt


def test_voter_prompt_uses_abc_labels_not_model_names():
    prompt = build_voter_prompt(
        question="Q",
        answer_a="A answer",
        answer_b="B answer",
        answer_c="C answer",
    )
    assert "ANSWER A" in prompt
    assert "ANSWER B" in prompt
    assert "ANSWER C" in prompt
    # Must not leak model identifiers
    assert "model_1" not in prompt
    assert "model_2" not in prompt
    assert "model_3" not in prompt


def test_voter_prompt_includes_output_format_instructions():
    prompt = build_voter_prompt("Q", "a1", "a2", "a3")
    assert "Vote:" in prompt
    assert "Confidence:" in prompt
    assert "Reason:" in prompt


def test_voter_prompt_handles_multiline_answers():
    multi = "Step 1: do this\nStep 2: do that\nStep 3: final answer is 42"
    prompt = build_voter_prompt("Solve it", multi, "42", "I don't know")
    assert "Step 1: do this" in prompt
    assert "Step 3: final answer is 42" in prompt


def test_voter_prompt_handles_code_answers():
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    prompt = build_voter_prompt("Write an add function", code, "def add(a,b): return a+b", "N/A")
    assert "def add" in prompt


def test_voter_prompt_handles_empty_answers():
    """Empty answers should not crash prompt building."""
    prompt = build_voter_prompt("Q", "", "", "")
    assert "ANSWER A" in prompt
    assert "ANSWER B" in prompt
    assert "ANSWER C" in prompt


# ---------------------------------------------------------------------------
# Aggregator prompt
# ---------------------------------------------------------------------------

def test_aggregator_prompt_contains_question():
    prompt = build_aggregator_prompt(
        question="What is 2 + 2?",
        answer_a="4",
        answer_b="5",
        answer_c="3",
        votes_summary="Model 1 voted: A\nModel 2 voted: A\nModel 3 voted: B",
    )
    assert "What is 2 + 2?" in prompt


def test_aggregator_prompt_contains_votes_summary():
    votes = "Model 1 voted: A (Confidence: High)\nModel 2 voted: B (Confidence: Low)"
    prompt = build_aggregator_prompt("Q", "a1", "a2", "a3", votes_summary=votes)
    assert "Model 1 voted: A" in prompt
    assert "Model 2 voted: B" in prompt


def test_aggregator_prompt_contains_final_answer_section():
    prompt = build_aggregator_prompt("Q", "a1", "a2", "a3", votes_summary="")
    assert "Final Answer:" in prompt
    assert "Winner:" in prompt


def test_aggregator_prompt_contains_all_answers():
    prompt = build_aggregator_prompt(
        question="Q",
        answer_a="alpha",
        answer_b="beta",
        answer_c="gamma",
        votes_summary="",
    )
    assert "alpha" in prompt
    assert "beta" in prompt
    assert "gamma" in prompt


# ---------------------------------------------------------------------------
# Domain-specific prompt content checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain,question", [
    ("math",    "What is the derivative of x^2?"),
    ("coding",  "Write a Python function to reverse a string."),
    ("logic",   "All men are mortal. Socrates is a man. Is Socrates mortal?"),
    ("english", "Rewrite this sentence in passive voice: The cat chased the mouse."),
])
def test_voter_prompt_embeds_domain_question(domain, question):
    """Voter prompt correctly embeds questions from all supported domains."""
    prompt = build_voter_prompt(question, "ans1", "ans2", "ans3")
    assert question in prompt
