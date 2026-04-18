from __future__ import annotations

import re

_VOTE_LABEL_RE = re.compile(r"\b([A-J])\b")

VOTER_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating answers to a question. "
    "Read each answer carefully, then select the single best one. "
    "Respond with exactly one letter (A, B, C, …) corresponding to the best answer. "
    "Do not explain your reasoning — output only the letter."
)

AGGREGATOR_SYSTEM_PROMPT = (
    "You are a tie-breaking arbitrator. "
    "The answers below are tied in votes. "
    "Select the single best answer and respond with exactly one letter. "
    "Do not explain — output only the letter."
)


def build_voter_prompt(question: str, labeled_answers: dict[str, str]) -> str:
    answers_block = "\n\n".join(
        f"Answer {label}:\n{text}" for label, text in labeled_answers.items()
    )
    return (
        f"Question:\n{question}\n\n"
        f"{answers_block}\n\n"
        "Which answer is best? Reply with a single letter only."
    )


def build_aggregator_prompt(
    question: str,
    labeled_answers: dict[str, str],
    vote_tally: dict[str, int],
) -> str:
    tally_note = ", ".join(f"{label}: {count} vote(s)" for label, count in vote_tally.items())
    answers_block = "\n\n".join(
        f"Answer {label}:\n{text}" for label, text in labeled_answers.items()
    )
    return (
        f"Question:\n{question}\n\n"
        f"{answers_block}\n\n"
        f"Current vote tally (tied): {tally_note}\n\n"
        "Break the tie. Reply with a single letter only."
    )


def parse_vote(response_text: str, valid_labels: list[str]) -> str | None:
    """Extract the first valid label letter from a vote response."""
    for match in _VOTE_LABEL_RE.finditer(response_text.strip()):
        label = match.group(1).upper()
        if label in valid_labels:
            return label
    return None
