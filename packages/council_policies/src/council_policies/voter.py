from __future__ import annotations

import asyncio
import logging
from collections import Counter

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator, OrchestratorResponse

from council_policies.prompts import (
    TIEBREAK_SYSTEM_PROMPT,
    VOTER_SYSTEM_PROMPT,
    build_tiebreak_prompt,
    build_voter_prompt,
    parse_vote,
)

logger = logging.getLogger(__name__)

_LABELS = list("ABCDEFGHIJ")


async def run_vote(
    question: str,
    candidates: dict[str, OrchestratorResponse],
    orchestrator: ModelOrchestrator,
    *,
    voter_roles: tuple[str, ...],
) -> tuple[str, dict[str, int]]:
    """
    Run a majority vote among voter_roles over the given candidate answers.

    Returns (winning_candidate_key, vote_tally_by_candidate_key).
    """
    keys = list(candidates)
    if len(keys) > len(_LABELS):
        raise ValueError(f"Too many candidates ({len(keys)}); max is {len(_LABELS)}")

    labeled_answers: dict[str, str] = {
        _LABELS[i]: candidates[k].text for i, k in enumerate(keys)
    }
    label_list = [_LABELS[i] for i in range(len(keys))]
    label_to_key = {_LABELS[i]: k for i, k in enumerate(keys)}

    voter_request = PromptRequest(
        system_prompt=VOTER_SYSTEM_PROMPT,
        user_prompt=build_voter_prompt(question, labeled_answers),
    )

    vote_tasks = [
        orchestrator.get_client(role).get_response(voter_request)
        for role in voter_roles
    ]
    results = await asyncio.gather(*vote_tasks, return_exceptions=True)

    tally: Counter[str] = Counter()
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Voter failed: %s", result)
            continue
        label = parse_vote(result.text, label_list)
        if label:
            tally[label_to_key[label]] += 1
        else:
            logger.debug("Could not parse vote from: %r", result.text[:100])

    if not tally:
        logger.warning("No valid votes cast; defaulting to first candidate")
        return keys[0], {}

    top_count = tally.most_common(1)[0][1]
    leaders = [k for k, v in tally.items() if v == top_count]

    if len(leaders) == 1:
        return leaders[0], dict(tally)

    # Tie-break: ask the default role (general) to arbitrate
    winning_label = await _break_tie(
        question=question,
        labeled_answers=labeled_answers,
        tied_labels=[_LABELS[keys.index(k)] for k in leaders],
        label_to_key=label_to_key,
        tally={_LABELS[keys.index(k)]: tally[k] for k in leaders},
        orchestrator=orchestrator,
    )
    winner = label_to_key.get(winning_label, leaders[0])
    return winner, dict(tally)


async def _break_tie(
    question: str,
    labeled_answers: dict[str, str],
    tied_labels: list[str],
    label_to_key: dict[str, str],
    tally: dict[str, int],
    orchestrator: ModelOrchestrator,
) -> str:
    tied_answers = {label: labeled_answers[label] for label in tied_labels}
    arb_request = PromptRequest(
        system_prompt=TIEBREAK_SYSTEM_PROMPT,
        user_prompt=build_tiebreak_prompt(question, tied_answers, tally),
    )
    try:
        result = await orchestrator.general_client.get_response(arb_request)
        label = parse_vote(result.text, tied_labels)
        if label:
            return label
    except Exception as exc:
        logger.warning("Tie-break arbitration failed: %s", exc)
    return tied_labels[0]
