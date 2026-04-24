"""
Unit tests for P2 council policy.

All tests are fully mocked — no real LLM calls or HuggingFace downloads.
Run with: pytest packages/council_policies/tests/test_p2.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from council_policies.models import PolicyEvaluationResult, UnifiedInput
from council_policies.p2.prompts import (
    LABEL_TO_ROLE,
    ROLE_TO_LABEL,
    build_synthesis_prompt,
    build_vote_prompt,
)
from council_policies.p2.policy import _majority, _parse_vote
from llm_gateway.models import Usage


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
    prompt = build_synthesis_prompt(
        source_dataset="quality",
        question="What is the capital?",
        context="France is a country in Europe.",
        winning_answer="Paris",
        other_answers=["paris", "PARIS"],
    )
    assert "What is the capital?" in prompt
    assert "France is a country in Europe." in prompt
    assert "Paris" in prompt
    assert "paris" in prompt


def test_synthesis_prompt_includes_dataset_specific_rules():
    fever_prompt = build_synthesis_prompt(
        source_dataset="fever",
        question="Claim?",
        context="Evidence",
        winning_answer="SUPPORTS",
        other_answers=["REFUTES"],
    )
    math_prompt = build_synthesis_prompt(
        source_dataset="hardmath",
        question="2+2?",
        context=None,
        winning_answer="\\boxed{4}",
        other_answers=["4"],
    )
    code_prompt = build_synthesis_prompt(
        source_dataset="humaneval_plus",
        question="def solve(): ...",
        context=None,
        winning_answer="def solve():\n    pass",
        other_answers=[],
    )

    assert "SUPPORTS, REFUTES, or NOT ENOUGH INFO" in fever_prompt
    assert "\\boxed{}" in math_prompt
    assert "Python code" in code_prompt


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
    resp.prompt_response = MagicMock(usage=Usage())
    return resp


async def test_generate_returns_winning_answer_and_metadata():
    """generate() runs answer→vote and returns the winning answer plus metadata."""
    from council_policies.adapter import P2PolicyClient
    from llm_gateway import PromptRequest

    async def mock_run(req, *, target=None, **kw):
        if "vote" in req.user_prompt.lower() or "Answer A:" in req.user_prompt:
            return _make_response("B")
        return _make_response(f"Answer from {target}.")

    mock_orch = MagicMock()
    mock_orch.run = mock_run

    client = P2PolicyClient(mock_orch)
    request = PromptRequest(user_prompt="What is the capital of France?", temperature=0.0)
    response = await client.generate(request)

    assert response.text == "Answer from reasoning."
    assert response.model == "p2_council"
    assert response.metadata["winning_label"] == "B"
    assert response.metadata["winning_role"] == "reasoning"
    assert response.metadata["winning_answer"] == "Answer from reasoning."
    assert response.metadata["winning_model"] == "reasoning"
    assert response.metadata["votes"] == {"qa": "B", "reasoning": "B", "general": "B"}
    assert response.metadata["role_results"]["reasoning"]["model"] == "reasoning"


async def test_generate_does_not_call_synthesis_stage():
    """The benchmark path should stop after voting and return the winner directly."""
    from council_policies.adapter import P2PolicyClient
    from llm_gateway import PromptRequest

    calls: list[tuple[str | None, str]] = []

    async def mock_run(req, *, target=None, **kw):
        calls.append((target, req.user_prompt))
        if "Answer A:" in req.user_prompt:
            return _make_response("B")  # all vote B
        return _make_response("The answer.")

    mock_orch = MagicMock()
    mock_orch.run = mock_run

    client = P2PolicyClient(mock_orch)
    request = PromptRequest(user_prompt="Q?", temperature=0.0)
    response = await client.generate(request)

    assert len(calls) == 6
    assert all("synth" not in prompt.lower() for _, prompt in calls)
    assert response.text == "The answer."
    assert response.model == "p2_council"


def test_extract_musique_oracle_context_prefers_supporting_paragraphs():
    from council_policies.p2.run import _extract_musique_oracle_context

    row = {
        "paragraphs": [
            {"paragraph_text": "ignore me", "is_supporting": False},
            {"paragraph_text": "support 1", "is_supporting": True},
            {"paragraph_text": "support 2", "is_supporting": True},
        ]
    }

    assert _extract_musique_oracle_context(row, None) == "support 1\n\nsupport 2"


def test_score_prediction_for_fever_uses_label_accuracy():
    from council_policies.p2.run import _score_prediction

    metric_name, primary_metric, values, metadata = _score_prediction(
        "SUPPORTS",
        {"source_dataset": "fever", "gold_label": "SUPPORTS"},
    )

    assert metric_name == "label_accuracy"
    assert primary_metric == "label_accuracy"
    assert values["label_accuracy"] == 1.0
    assert metadata["extracted_answer"] == "SUPPORTS"


def test_score_prediction_for_musique_uses_final_answer_format():
    from council_policies.p2.run import _score_prediction

    metric_name, primary_metric, values, metadata = _score_prediction(
        "Final Answer: Joe Cabot",
        {
            "source_dataset": "musique",
            "gold_answer": "Joe Cabot",
            "parsed_metadata": {"answerable": True},
        },
    )

    assert metric_name == "musique_token_f1"
    assert primary_metric == "token_f1"
    assert values["token_f1"] == 1.0
    assert metadata["answerable"] is True


def test_score_records_macro_average_datasets():
    from council_policies.p2.run import _score_synthesized_records

    score_records, dataset_scores, combined_metric = _score_synthesized_records([
        {
            "example_id": "1",
            "dataset_name": "router_dataset",
            "source_dataset": "fever",
            "status": "success",
            "metric_name": "label_accuracy",
            "synthesized_text": "SUPPORTS",
            "metadata": {"row_metadata": {"source_dataset": "fever", "gold_label": "SUPPORTS"}},
        },
        {
            "example_id": "2",
            "dataset_name": "router_dataset",
            "source_dataset": "quality",
            "status": "success",
            "metric_name": "token_f1",
            "synthesized_text": "Paris",
            "metadata": {"row_metadata": {"source_dataset": "quality", "gold_answer": "Paris"}},
        },
    ])

    assert len(score_records) == 2
    assert dataset_scores["fever"] == 1.0
    assert dataset_scores["quality"] == 1.0
    assert combined_metric == 1.0


def test_build_row_system_metrics_exposes_specialists_and_excludes_synth_from_final_totals():
    from council_policies.p2.run import _build_row_system_metrics

    specialist = {
        "system_metrics": {
            "answer_usage": {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45},
            "vote_usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
            "specialist_usage": {"input_tokens": 42, "output_tokens": 18, "total_tokens": 60},
            "answer_latency_ms_total": 600.0,
            "vote_latency_ms_total": 180.0,
            "specialist_latency_ms_total": 780.0,
            "wall_clock_latency_ms": 300.0,
        },
        "role_results": {
            "qa": {"latency_ms": 100.0, "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            "reasoning": {"latency_ms": 200.0, "usage": {"input_tokens": 20, "output_tokens": 7, "total_tokens": 27}},
            "general": {"latency_ms": 300.0, "usage": {"input_tokens": 0, "output_tokens": 3, "total_tokens": 3}},
        },
    }
    synth = {
        "usage": {"input_tokens": 999, "output_tokens": 111, "total_tokens": 1110},
        "latency_ms": 2222.0,
    }

    metrics = _build_row_system_metrics(specialist, synth)

    assert metrics["qa_usage"]["total_tokens"] == 15
    assert metrics["reasoning_usage"]["total_tokens"] == 27
    assert metrics["general_usage"]["total_tokens"] == 3
    assert metrics["qa_latency_ms"] == 100.0
    assert metrics["reasoning_latency_ms"] == 200.0
    assert metrics["general_latency_ms"] == 300.0
    assert metrics["specialist_usage_total"]["total_tokens"] == 60
    assert metrics["final_row_usage"]["total_tokens"] == 60
    assert metrics["specialist_latency_ms_total"] == 780.0
    assert metrics["final_row_latency_ms"] == 780.0
    assert metrics["synthesizer_usage"]["total_tokens"] == 1110
    assert metrics["synthesizer_latency_ms"] == 2222.0


def test_prediction_records_use_specialist_only_top_level_metrics():
    from council_policies.p2.run import _build_prediction_records
    from council_policies.models import P2RunConfig
    from pathlib import Path

    specialist = {
        "example_id": "1",
        "status": "success",
        "request": {},
        "row_metadata": {"source_dataset": "fever"},
        "system_metrics": {
            "answer_usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            "vote_usage": {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
            "specialist_usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
            "answer_latency_ms_total": 10.0,
            "vote_latency_ms_total": 20.0,
            "specialist_latency_ms_total": 30.0,
            "wall_clock_latency_ms": 15.0,
        },
        "role_results": {
            "qa": {"latency_ms": 1.0, "usage": {"total_tokens": 1}},
            "reasoning": {"latency_ms": 2.0, "usage": {"total_tokens": 2}},
            "general": {"latency_ms": 3.0, "usage": {"total_tokens": 3}},
        },
    }
    synth = {
        "example_id": "1",
        "status": "success",
        "synthesized_text": "SUPPORTS",
        "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
        "latency_ms": 400.0,
        "request_id": "req-1",
        "finish_reason": "stop",
        "provider": "test",
    }
    score = {"example_id": "1", "status": "scored"}
    config = P2RunConfig(output_root=Path("/tmp"))

    rows = _build_prediction_records(
        run_id="run-1",
        config=config,
        specialist_records=[specialist],
        synthesized_records=[synth],
        score_records=[score],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["usage"]["total_tokens"] == 12
    assert row["latency_ms"] == 30.0
    assert row["response_metadata"]["system_metrics"]["synthesizer_usage"]["total_tokens"] == 300
    assert row["response_metadata"]["system_metrics"]["synthesizer_latency_ms"] == 400.0
