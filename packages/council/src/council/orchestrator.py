from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import List, Tuple

from llm_gateway.models import PromptRequest
from llm_gateway.base import BaseLLMClient


# -------------------------
# Config (tunable knobs)
# -------------------------
DEFAULT_TIMEOUT = 20
MIN_RESPONSES = 2
MAX_MODELS = 5


# -------------------------
# Normalize responses
# -------------------------
def _normalize(text: str) -> str:
    return text.strip().lower().replace("\n", " ")


# -------------------------
# Clean responses (NEW)
# -------------------------
def _clean_response(text: str) -> str:
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove excessive whitespace
    return text.strip()


# -------------------------
# Extract vote safely
# -------------------------
def _extract_vote(text: str) -> int | None:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


# -------------------------
# Majority detection
# -------------------------
def _get_majority(responses: List[str]) -> Tuple[str | None, float]:
    counts = Counter(_normalize(r) for r in responses)
    top_norm, count = counts.most_common(1)[0]

    confidence = count / len(responses)

    if count >= (len(responses) // 2 + 1):
        for r in responses:
            if _normalize(r) == top_norm:
                return r, confidence

    return None, confidence


# -------------------------
# Build vote prompt
# -------------------------
def _build_vote_prompt(original_prompt: str, responses: List[str]) -> str:
    formatted = "\n".join(
        [f"{i+1}. {resp}" for i, resp in enumerate(responses)]
    )

    return f"""
You are an impartial judge.

Your job is to evaluate multiple candidate answers.

Rules:
- Do NOT favor your own previous answers
- Be unbiased
- Focus only on correctness

Question:
{original_prompt}

Candidate answers:
{formatted}

Return ONLY the number (1-{len(responses)})
"""


# -------------------------
# Safe async with timeout
# -------------------------
async def _safe_generate(client, request):
    try:
        result = await asyncio.wait_for(
            client.generate(request),
            timeout=DEFAULT_TIMEOUT,
        )

        cleaned = _clean_response(result.text)

        print(f"[SUCCESS] {client.provider}: {cleaned}")

        return cleaned if cleaned else None

    except Exception as e:
        print(f"[ERROR] {client.provider}: {e}")
        return None


# -------------------------
# Generate from all clients
# -------------------------
async def _generate_all(
    clients: List[BaseLLMClient],
    request: PromptRequest,
) -> List[str]:

    tasks = [
        asyncio.create_task(_safe_generate(client, request))
        for client in clients[:MAX_MODELS]
    ]

    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


# -------------------------
# MAIN: Council loop
# -------------------------
async def council_generate(
    clients: List[BaseLLMClient],
    request: PromptRequest,
    *,
    max_rounds: int = 3,
    min_confidence: float = 0.6,
    debug: bool = False,
) -> str:

    responses = await _generate_all(clients, request)

    if len(responses) < MIN_RESPONSES:
        return responses[0] if responses else "No valid responses"

    for round_idx in range(max_rounds):

        if debug:
            print(f"\n[Round {round_idx+1}] Responses:", responses)

        # Majority check
        majority, confidence = _get_majority(responses)

        if majority and confidence >= min_confidence:
            if debug:
                print("Majority reached:", majority)
            return majority

        # Voting round
        vote_prompt = _build_vote_prompt(
            request.user_prompt or "",
            responses,
        )

        vote_request = PromptRequest(user_prompt=vote_prompt)

        vote_outputs = await _generate_all(clients, vote_request)

        votes: List[int] = []
        for v in vote_outputs:
            idx = _extract_vote(v)
            if idx is not None:
                votes.append(idx)

        if not votes:
            return responses[0]

        vote_counts = Counter(votes)
        best_idx = vote_counts.most_common(1)[0][0] - 1

        if not (0 <= best_idx < len(responses)):
            return responses[0]

        chosen = responses[best_idx]

        if debug:
            print("Votes:", votes)
            print("Chosen:", chosen)

        # Soft convergence
        new_responses = []

        for v in votes:
            idx = v - 1
            if 0 <= idx < len(responses):
                new_responses.append(responses[idx])
            else:
                new_responses.append(chosen)

        responses = new_responses

    return responses[0]