from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Voter (P2): select from redundant attempts at the same task.
# Each LLM in the flat council sees the same prompt; the voter picks one.
# --------------------------------------------------------------------------- #


_VOTE_LABEL_RE = re.compile(r"\b([A-J])\b")


VOTER_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating answers to a question. "
    "Read each answer carefully, then select the single best one. "
    "Respond with exactly one letter (A, B, C, …) corresponding to the best answer. "
    "Do not explain your reasoning — output only the letter."
)


TIEBREAK_SYSTEM_PROMPT = (
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


def build_tiebreak_prompt(
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


# --------------------------------------------------------------------------- #
# Synthesizer (P3/P4 multi-skill): combine complementary specialist outputs.
# Each specialist handled a different slice of the task; the synthesizer fuses
# them into one answer without dropping any claim.
# --------------------------------------------------------------------------- #


SYNTHESIZER_SYSTEM_PROMPT = (
    "You are a synthesizer. You receive one or more partial responses from "
    "specialist models, each labeled with its role. Combine them into a "
    "single coherent answer to the user's question. "
    "Preserve every claim from every partial — do not drop, discard, or "
    "override any specialist's contribution. Merge overlapping information "
    "without redundancy. If two partials disagree, include both positions "
    "and attribute them to their roles. Do not introduce new claims that "
    "are not present in the partials, and do not add analysis of your own. "
    "Output only the synthesized answer."
)


def build_synthesis_prompt(
    question: str,
    partials: dict[str, str],
) -> str:
    """
    Format the synthesizer's user prompt from per-role specialist outputs.

    `partials` maps role name -> full response text (not extracted answer —
    the synthesizer gets the reasoning too).
    """
    blocks = [
        f"The {role} specialist produced the following response:\n{text}"
        for role, text in partials.items()
    ]
    joined = "\n\n---\n\n".join(blocks)
    return (
        f"Original question:\n{question}\n\n"
        f"{joined}\n\n"
        "Synthesize the above into a single final answer. "
        "Preserve every claim from every specialist."
    )
