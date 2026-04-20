"""
Unit tests for P2 council policy.

All tests are fully mocked — no real LLM calls or HuggingFace downloads.
Run with: pytest packages/council_policies/tests/test_p2.py -v
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from council_policies.models import PolicyEvaluationResult, UnifiedInput
from council_policies.p2.prompts import (
    LABEL_TO_ROLE,
    ROLE_TO_LABEL,
    build_synthesis_prompt,
    build_vote_prompt,
)
from council_policies.p2.policy import _majority, _parse_vote


# ── Prompt builders ───────────────────────────────────────────────────────────

def test_role_to_label_covers_all_specialists():
    assert set(ROLE_TO_LABEL.keys()) == {"qa", "reasoning", "general"}
    assert set(ROLE_TO_LABEL.values()) == {"A", "B", "C"}


def test_label_to_role_is_inverse():
    for role, label in ROLE_TO_LABEL.items():
        assert LABEL_TO_ROLE[label] == role


def test_vote_prompt_contains_question_and_answers():
    prompt = build_vote_prompt("What is 2+2?", {"A": "four", "B": "4", "C": "Two plus two"})
    assert "What is 2+2?" in prompt
    assert "Answer A:" in prompt
    assert "four" in prompt
    assert "4" in prompt
    assert "A, B, or C" in prompt


def test_synthesis_prompt_contains_question_and_answers():
    prompt = build_synthesis_prompt("What is the capital?", "Paris", ["paris", "PARIS"])
    assert "What is the capital?" in prompt
    assert "Paris" in prompt
    assert "paris" in prompt


# ── Vote parsing ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("A", "A"),
    ("The answer is B.", "B"),
    ("  c  ", "C"),
    ("I choose B because...", "B"),
    ("nonsense", "A"),
    ("", "A"),
])
def test_parse_vote(text, expected):
    assert _parse_vote(text) == expected


# ── Majority vote ─────────────────────────────────────────────────────────────

def test_majority_clear_winner():
    assert _majority({"qa": "B", "reasoning": "B", "general": "A"}) == "B"


def test_majority_unanimous():
    assert _majority({"qa": "C", "reasoning": "C", "general": "C"}) == "C"


def test_majority_tie_breaks_alphabetically():
    # All three differ — A wins alphabetically
    assert _majority({"qa": "A", "reasoning": "B", "general": "C"}) == "A"


# ── Data models ───────────────────────────────────────────────────────────────

def test_unified_input_fields():
    inp = UnifiedInput(
        example_id="ex_1",
        dataset_name="router_dataset",
        question="Who wrote Hamlet?",
    )
    assert inp.example_id == "ex_1"
    assert inp.question == "Who wrote Hamlet?"


def test_policy_evaluation_result_success():
    result = PolicyEvaluationResult(
        example_id="ex_1",
        dataset_name="router_dataset",
        output="Shakespeare",
        status="success",
        metadata={"winning_label": "B", "winning_answer": "Shakespeare"},
    )
    assert result.output == "Shakespeare"
    assert result.status == "success"
    assert result.error_type is None


def test_policy_evaluation_result_error():
    result = PolicyEvaluationResult(
        example_id="ex_2",
        dataset_name="router_dataset",
        output="",
        status="error",
        error_type="TimeoutError",
        error_message="Request timed out",
    )
    assert result.status == "error"
    assert result.error_type == "TimeoutError"


# ── P2PolicyClient.generate() (mocked orchestrator) ──────────────────────────

def _make_response(text: str):
    resp = MagicMock()
    resp.text = text
    return resp


async def test_generate_returns_synthesized_answer():
    """generate() runs answer→vote→synthesize and returns the final text."""
    from council_policies.adapter import P2PolicyClient
    from llm_gateway import PromptRequest

    call_num = [0]

    async def mock_run(req, *, target=None, **kw):
        call_num[0] += 1
        # Synthesize call (last call, target=general, temperature=0.3 in prompt)
        if "improve" in req.user_prompt.lower() or "synthesize" in req.user_prompt.lower():
            return _make_response("Final synthesized answer.")
        # Vote calls return B
        if "vote" in req.user_prompt.lower() or "Answer A:" in req.user_prompt:
            return _make_response("B")
        # Answer calls
        return _make_response(f"Answer from {target}.")

    mock_orch = MagicMock()
    mock_orch.run = mock_run

    client = P2PolicyClient(mock_orch)
    request = PromptRequest(user_prompt="What is the capital of France?", temperature=0.0)
    response = await client.generate(request)

    assert response.text == "Final synthesized answer."
    assert response.model == "p2_council"


async def test_generate_falls_back_to_winner_on_synthesis_error():
    """If synthesis fails, generate() falls back to the voting winner's answer."""
    from council_policies.adapter import P2PolicyClient
    from llm_gateway import PromptRequest

    async def mock_run(req, *, target=None, **kw):
        if "improve" in req.user_prompt.lower() or "synthesize" in req.user_prompt.lower():
            raise RuntimeError("synthesis error")
        if "Answer A:" in req.user_prompt:
            return _make_response("B")  # all vote B
        return _make_response("The answer.")  # all models give same answer

    mock_orch = MagicMock()
    mock_orch.run = mock_run

    client = P2PolicyClient(mock_orch)
    request = PromptRequest(user_prompt="Q?", temperature=0.0)
    response = await client.generate(request)

    assert response.text != ""  # falls back to winning answer
    assert response.model == "p2_council"
