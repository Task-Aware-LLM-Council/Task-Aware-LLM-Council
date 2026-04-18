from __future__ import annotations

import asyncio
import logging
from collections import Counter

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator, OrchestratorResponse

from council_policies2.prompts import (
    AGGREGATOR_SYSTEM_PROMPT,
    VOTER_SYSTEM_PROMPT,
    build_aggregator_prompt,
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
    arbitrator_role: str = "general",
) -> tuple[str, dict[str, int]]:
    """
    Run a majority vote among ``voter_roles`` over the given candidate answers.

    Parameters
    ----------
    question:
        The original question text shown to voters.
    candidates:
        Mapping from role-name → OrchestratorResponse.  Each value becomes one
        labeled candidate; each key is used as a voter.
    orchestrator:
        The active ModelOrchestrator.
    voter_roles:
        The roles whose responses will cast votes.  Normally the same as
        ``candidates.keys()``, but a subset is fine (e.g. after one member fails).
    arbitrator_role:
        The role used to break ties.  Defaults to ``"general"``.  If the role is
        not registered in the orchestrator the tie-break is skipped gracefully and
        the first tied candidate wins.

    Returns
    -------
    (winning_candidate_key, vote_tally_by_candidate_key)
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

    # FIX (bug 5): voter_roles may be smaller than the original council when
    # some members failed.  This is intentional — surviving members vote over
    # surviving answers — but is now documented here explicitly.
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

    # Tie-break: ask the arbitrator role to choose.
    winning_label = await _break_tie(
        question=question,
        labeled_answers=labeled_answers,
        tied_labels=[_LABELS[keys.index(k)] for k in leaders],
        label_to_key=label_to_key,
        tally={_LABELS[keys.index(k)]: tally[k] for k in leaders},
        orchestrator=orchestrator,
        arbitrator_role=arbitrator_role,
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
    *,
    arbitrator_role: str,
) -> str:
    """
    Ask ``arbitrator_role`` to break a vote tie.

    FIX (bug 3): The arbitrator role is now a parameter rather than the
    hardcoded string ``"general"``.  If the role is not registered in the
    orchestrator the KeyError is caught and the first tied label is returned as
    a safe fallback.
    """
    tied_answers = {label: labeled_answers[label] for label in tied_labels}
    arb_request = PromptRequest(
        system_prompt=AGGREGATOR_SYSTEM_PROMPT,
        user_prompt=build_aggregator_prompt(question, tied_answers, tally),
    )
    try:
        result = await orchestrator.get_client(arbitrator_role).get_response(arb_request)
        label = parse_vote(result.text, tied_labels)
        if label:
            return label
    except KeyError:
        # FIX (bug 3): arbitrator role not configured — log and fall through.
        logger.warning(
            "Tie-break arbitrator role %r is not registered in the orchestrator; "
            "defaulting to first tied candidate.",
            arbitrator_role,
        )
    except Exception as exc:
        logger.warning("Tie-break arbitration failed: %s", exc)
    return tied_labels[0]
