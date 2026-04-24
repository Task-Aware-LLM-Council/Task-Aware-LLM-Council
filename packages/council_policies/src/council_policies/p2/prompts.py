from __future__ import annotations

# Maps each specialist role to an anonymous answer label shown during voting.
# This hides which model produced which answer so models can't vote for themselves.
ROLE_TO_LABEL: dict[str, str] = {"qa": "A", "reasoning": "B", "general": "C"}
LABEL_TO_ROLE: dict[str, str] = {v: k for k, v in ROLE_TO_LABEL.items()}


def build_vote_prompt(question: str, label_to_answer: dict[str, str]) -> str:
    """Ask a model to pick the best answer from A, B, C."""
    answer_block = "\n\n".join(
        f"Answer {label}:\n{text}" for label, text in sorted(label_to_answer.items())
    )
    return (
        "You are evaluating three candidate answers to a question. "
        "Pick the most accurate and complete one.\n\n"
        f"Question: {question}\n\n"
        f"{answer_block}\n\n"
        "Which answer is best? Reply with only the letter A, B, or C."
    )


def build_synthesis_prompt(
    *,
    source_dataset: str,
    question: str,
    context: str | None,
    winning_answer: str,
    other_answers: list[str],
) -> str:
    """Ask the synthesizer to normalize the voted answer for the source dataset."""
    others = "\n\n".join(f"- {a}" for a in other_answers)
    context_block = f"Context:\n{context}\n\n" if context else ""
    dataset_rules = _dataset_format_rules(source_dataset)
    return (
        "You are the council synthesizer. Your job is to standardize the voted answer so it matches the exact scoring format for the source dataset.\n\n"
        "STRICT RULES:\n"
        "1. Output ONLY the final normalized answer.\n"
        "2. No explanation, no analysis, no bullet points, no commentary.\n"
        "3. Start from the PRIMARY answer, but fix formatting or incorporate correct details from the other answers if needed.\n"
        "4. Follow the dataset-specific output contract exactly.\n\n"
        f"Source dataset: {source_dataset}\n"
        f"Dataset-specific output contract:\n{dataset_rules}\n\n"
        f"Question: {question}\n\n"
        f"{context_block}"
        f"Primary answer (voted best):\n{winning_answer}\n\n"
        f"Other candidate answers:\n{others}\n\n"
        "Normalized final answer:"
    )


def _dataset_format_rules(source_dataset: str) -> str:
    normalized = (source_dataset or "").strip().lower()
    if "musique" in normalized:
        return (
            "Output the answer on a new line in the exact format: Final Answer: <answer>\n"
            "If the answer is missing from context, output exactly: Final Answer: NOT PRESENT IN CONTEXT"
        )
    if "fever" in normalized:
        return "Output exactly one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO"
    if "hardmath" in normalized or "math" in normalized:
        return "Output only the final answer enclosed in \\boxed{}"
    if "humaneval" in normalized or "code" in normalized:
        return "Output only executable Python code. No prose, no markdown fences."
    return "Output only a concise direct answer with no explanation."
