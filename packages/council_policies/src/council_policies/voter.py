"""
Council voting logic.

Flow:
  1. Send question to all 3 models → collect answers
  2. Send all 3 answers back to all 3 models (anonymized as A/B/C)
  3. Each model votes for the best answer
  4. Tally votes → majority wins
  5. If tied (1-1-1), run aggregator prompt with a single arbitrator model
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from llm_gateway.factory import create_client
from llm_gateway.models import PromptRequest, ProviderConfig

from council_policies.prompts import build_aggregator_prompt, build_voter_prompt


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Identifies one model in the council."""
    model_id: str        # e.g. "openai/gpt-4o-mini"
    provider_config: ProviderConfig


@dataclass
class CouncilAnswer:
    model_config: ModelConfig
    answer: str


@dataclass
class Vote:
    voter_model_id: str
    voted_for: str       # "A", "B", or "C"
    confidence: str      # "High", "Medium", "Low"
    reason: str


@dataclass
class CouncilResult:
    question: str
    answers: list[CouncilAnswer]          # 3 answers in order (index 0 = A, 1 = B, 2 = C)
    votes: list[Vote]                     # 3 votes, one per model
    winner_label: str                     # "A", "B", or "C" (or "NONE" if aggregator says so)
    winning_answer: str                   # the actual answer text
    winning_model_id: str                 # which model produced the winning answer
    tiebreak_used: bool = False           # True if aggregator was needed


# ---------------------------------------------------------------------------
# Step 1: Ask each model the question
# ---------------------------------------------------------------------------

async def _get_answer(model_cfg: ModelConfig, question: str) -> CouncilAnswer:
    client = create_client(model_cfg.provider_config)
    request = PromptRequest(
        model=model_cfg.model_id,
        user_prompt=question,
    )
    async with client:
        response = await client.generate(request)
    return CouncilAnswer(model_config=model_cfg, answer=response.text.strip())


async def gather_answers(models: list[ModelConfig], question: str) -> list[CouncilAnswer]:
    """Send the question to all models in parallel and collect answers.

    If a model fails, its answer is recorded as an empty string so the
    council can still proceed with the remaining answers.
    """
    tasks = [_get_answer(m, question) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    answers: list[CouncilAnswer] = []
    for model, result in zip(models, results):
        if isinstance(result, BaseException):
            answers.append(CouncilAnswer(model_config=model, answer=""))
        else:
            answers.append(result)
    return answers


# ---------------------------------------------------------------------------
# Step 2: Send all answers back to each model for voting
# ---------------------------------------------------------------------------

def _parse_vote(raw: str, voter_model_id: str) -> Vote:
    """Extract Vote/Confidence/Reason from model output."""
    vote_match = re.search(r"Vote:\s*([ABC])", raw, re.IGNORECASE)
    conf_match = re.search(r"Confidence:\s*(High|Medium|Low)", raw, re.IGNORECASE)
    reason_match = re.search(r"Reason:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)

    voted_for = vote_match.group(1).upper() if vote_match else "A"
    confidence = conf_match.group(1).capitalize() if conf_match else "Low"
    reason = reason_match.group(1).strip() if reason_match else raw.strip()

    return Vote(
        voter_model_id=voter_model_id,
        voted_for=voted_for,
        confidence=confidence,
        reason=reason,
    )


async def _cast_vote(
    voter: ModelConfig,
    question: str,
    answer_a: str,
    answer_b: str,
    answer_c: str,
) -> Vote:
    client = create_client(voter.provider_config)
    prompt = build_voter_prompt(question, answer_a, answer_b, answer_c)
    request = PromptRequest(
        model=voter.model_id,
        user_prompt=prompt,
        temperature=0.0,   # deterministic voting
    )
    async with client:
        response = await client.generate(request)
    return _parse_vote(response.text, voter.model_id)


async def gather_votes(
    models: list[ModelConfig],
    question: str,
    answers: list[CouncilAnswer],
) -> list[Vote]:
    """Send all 3 answers back to all 3 models and collect votes in parallel.

    If a model fails to vote, it is recorded as abstaining (voted_for="A",
    confidence="Low") so the tally can still proceed.
    """
    a, b, c = answers[0].answer, answers[1].answer, answers[2].answer
    tasks = [_cast_vote(m, question, a, b, c) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    votes: list[Vote] = []
    for model, result in zip(models, results):
        if isinstance(result, BaseException):
            votes.append(Vote(
                voter_model_id=model.model_id,
                voted_for="A",
                confidence="Low",
                reason=f"Vote failed: {type(result).__name__}: {result}",
            ))
        else:
            votes.append(result)
    return votes


# ---------------------------------------------------------------------------
# Step 3: Tally votes
# ---------------------------------------------------------------------------

def tally_votes(votes: list[Vote]) -> dict[str, int]:
    tally: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for v in votes:
        if v.voted_for in tally:
            tally[v.voted_for] += 1
    return tally


def majority_winner(tally: dict[str, int]) -> str | None:
    """Return the winner label if any answer has >= 2 votes, else None (tie)."""
    best_label = max(tally, key=lambda k: tally[k])
    if tally[best_label] >= 2:
        return best_label
    return None  # 1-1-1 tie


# ---------------------------------------------------------------------------
# Step 4 (optional): Tiebreak via aggregator
# ---------------------------------------------------------------------------

def _parse_aggregator_result(raw: str) -> tuple[str, str | None]:
    """Returns (winner_label, final_answer_or_None)."""
    winner_match = re.search(r"Winner:\s*([ABC]|NONE)", raw, re.IGNORECASE)
    winner = winner_match.group(1).upper() if winner_match else "NONE"
    final_match = re.search(r"Final Answer:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    final_answer = final_match.group(1).strip() if final_match else None
    return winner, final_answer


async def run_aggregator(
    arbitrator: ModelConfig,
    question: str,
    answers: list[CouncilAnswer],
    votes: list[Vote],
) -> tuple[str, str | None]:
    """Call the aggregator prompt with one model to break a tie.
    Returns (winning_label, final_answer) where final_answer may override the winning answer.
    """
    votes_summary = "\n".join(
        f"{v.voter_model_id} voted: {v.voted_for} (Confidence: {v.confidence})"
        for v in votes
    )
    prompt = build_aggregator_prompt(
        question=question,
        answer_a=answers[0].answer,
        answer_b=answers[1].answer,
        answer_c=answers[2].answer,
        votes_summary=votes_summary,
    )
    client = create_client(arbitrator.provider_config)
    request = PromptRequest(
        model=arbitrator.model_id,
        user_prompt=prompt,
        temperature=0.0,
    )
    async with client:
        response = await client.generate(request)
    return _parse_aggregator_result(response.text)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_council(
    models: list[ModelConfig],
    question: str,
    arbitrator: ModelConfig | None = None,
) -> CouncilResult:
    """
    Full council flow:
      1. Ask question to all 3 models
      2. Send all 3 answers back to all 3 models for voting
      3. Tally votes → majority wins
      4. If tied, use arbitrator (defaults to models[0])

    Args:
        models:      Exactly 3 ModelConfig instances.
        question:    The question to ask.
        arbitrator:  Model used for tiebreaking. Defaults to models[0].

    Returns:
        CouncilResult with the winner and all intermediate data.
    """
    if len(models) != 3:
        raise ValueError(f"Council requires exactly 3 models, got {len(models)}")
    if not question or not question.strip():
        raise ValueError("Question must not be empty")

    # Step 1: collect answers
    answers = await gather_answers(models, question)

    # Step 2: each model votes
    votes = await gather_votes(models, question, answers)

    # Step 3: tally
    tally = tally_votes(votes)
    winner_label = majority_winner(tally)
    tiebreak_used = False

    # Step 4: tiebreak if needed
    aggregator_final_answer: str | None = None
    if winner_label is None:
        tiebreak_used = True
        arb = arbitrator or models[0]
        winner_label, aggregator_final_answer = await run_aggregator(arb, question, answers, votes)

    # Map label back to answer/model
    label_to_index = {"A": 0, "B": 1, "C": 2}
    idx = label_to_index.get(winner_label, 0)
    winning_answer_obj = answers[idx] if winner_label != "NONE" else answers[0]
    winning_answer = aggregator_final_answer or winning_answer_obj.answer

    return CouncilResult(
        question=question,
        answers=answers,
        votes=votes,
        winner_label=winner_label,
        winning_answer=winning_answer,
        winning_model_id=winning_answer_obj.model_config.model_id,
        tiebreak_used=tiebreak_used,
    )
