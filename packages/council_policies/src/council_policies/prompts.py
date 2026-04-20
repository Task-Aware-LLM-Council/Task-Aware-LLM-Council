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
# Rater (P2 dataset council): score each candidate answer numerically.
# Same labeled layout as the voter prompt but asks for JSON scores 1-10
# instead of a single letter, so ratings can be aggregated across questions.
# --------------------------------------------------------------------------- #

_RATING_JSON_SCHEMA = (
    '{"A": {"score": <1-10>, "reasoning": "<one sentence>"},'
    ' "B": {"score": <1-10>, "reasoning": "<one sentence>"},'
    ' "C": {"score": <1-10>, "reasoning": "<one sentence>"}}'
)

RATER_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating answers to a question. "
    "Score each answer from 1 to 10 based on accuracy, completeness, and clarity. "
    "10 is the best possible answer; 1 is completely wrong or irrelevant. "
    "Do NOT factor in answer length alone. "
    "Output ONLY a JSON object — no prose before or after it."
)


def build_rating_prompt(question: str, labeled_answers: dict[str, str]) -> str:
    """
    Reuses the same labeled-answers layout as build_voter_prompt but instructs
    the model to return JSON scores instead of a single letter.
    """
    answers_block = "\n\n".join(
        f"Answer {label}:\n{text}" for label, text in labeled_answers.items()
    )
    return (
        f"Question:\n{question}\n\n"
        f"{answers_block}\n\n"
        f"Rate every answer using exactly this JSON format:\n{_RATING_JSON_SCHEMA}"
    )


def parse_ratings(
    response_text: str, valid_labels: list[str]
) -> dict[str, float] | None:
    """
    Parse a JSON rating response into a label→score dict.
    Tries strict JSON first, then a regex fallback.
    Returns None if neither yields all expected labels.
    """
    import json
    import re

    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", response_text).strip()

    def _clamp(v: float) -> float:
        return max(1.0, min(10.0, v))

    # Attempt 1: full JSON
    try:
        data = json.loads(cleaned)
        scores: dict[str, float] = {}
        for label in valid_labels:
            if label not in data:
                return None
            scores[label] = _clamp(float(data[label]["score"]))
        return scores
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # Attempt 2: regex per label
    scores = {}
    for label in valid_labels:
        m = re.search(
            rf'"{label}"\s*:\s*\{{[^}}]*"score"\s*:\s*(\d+(?:\.\d+)?)', cleaned
        )
        if not m:
            return None
        scores[label] = _clamp(float(m.group(1)))
    return scores if len(scores) == len(valid_labels) else None


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


# --------------------------------------------------------------------------- #
# Ordered synthesizer (P4 multi-run): combine partials in subtask order.
# Different from `build_synthesis_prompt` above because ordered synthesis
# carries run ordering across multiple potential same-role runs (the
# non-adjacent [A, B, A] case) — the synthesizer must be told to preserve
# the step sequence, not just merge claims.
# --------------------------------------------------------------------------- #


ORDERED_SYNTHESIZER_SYSTEM_PROMPT = (
    "You are a synthesizer that composes a final answer from a sequence of "
    "ordered partial responses. Each partial is labeled with a step index "
    "and the specialist role that produced it. "
    "Combine them into a single coherent answer that preserves the step "
    "sequence — later steps may depend on earlier ones, and reordering "
    "them can change the meaning. "
    "Preserve every claim from every partial. Merge overlapping information "
    "without redundancy. If two partials disagree, include both positions "
    "and attribute them to their step + role. Do not introduce new claims "
    "that are not present in the partials, and do not add analysis of your "
    "own. Output only the synthesized answer."
)


def build_ordered_synthesis_prompt(
    question: str,
    ordered_partials: list[tuple[int, str, str]],
) -> str:
    """
    Format the ordered synthesizer's user prompt.

    `ordered_partials` is a list of `(step_index, role, response_text)`
    tuples in execution order. Keeping the shape as a positional tuple
    (not a dict) is deliberate — the same role can appear more than once
    in the sequence, which a dict would collapse.
    """
    blocks = [
        f"Step {idx + 1} — {role} specialist:\n{text}"
        for idx, role, text in ordered_partials
    ]
    joined = "\n\n---\n\n".join(blocks)
    return (
        f"Original question:\n{question}\n\n"
        f"{joined}\n\n"
        "Synthesize the above into a single final answer. "
        "Preserve every claim and the step sequence."
    )
