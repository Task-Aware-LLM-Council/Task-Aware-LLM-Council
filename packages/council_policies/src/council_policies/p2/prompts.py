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
    question: str,
    winning_answer: str,
    other_answers: list[str],
) -> str:
    """Ask the synthesizer to write a final improved answer."""
    others = "\n\n".join(f"- {a}" for a in other_answers)
    return (
        "You are synthesizing a final answer from multiple candidates. "
        "Use the primary answer as your foundation and improve it with any correct details from the others.\n\n"
        "IMPORTANT: If the primary answer contains code, return it exactly as-is without any modifications.\n\n"
        f"Question: {question}\n\n"
        f"Primary answer:\n{winning_answer}\n\n"
        f"Other candidate answers:\n{others}\n\n"
        "Write the best possible final answer:"
    )
