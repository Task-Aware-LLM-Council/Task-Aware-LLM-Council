from __future__ import annotations

# Maps each specialist role to an anonymous answer label shown during voting.
# This hides which model produced which answer so models can't vote for themselves.
ROLE_TO_LABEL: dict[str, str] = {"qa": "A", "reasoning": "B", "general": "C"}
LABEL_TO_ROLE: dict[str, str] = {v: k for k, v in ROLE_TO_LABEL.items()}
_MUSIQUE_NOT_PRESENT = "NOT PRESENT IN CONTEXT"


def build_specialist_prompt(
    *,
    source_dataset: str,
    question: str,
    context: str | None,
    skill_tags: list[str] | None = None,
) -> tuple[str, str | None]:
    """Build the specialist prompt using the same dataset prompt shapes as task_eval profiles."""
    skill_tags = skill_tags or []
    normalized = (source_dataset or "").strip().lower()

    if "musique" in normalized:
        prompt = (
            "You are a strict reading comprehension assistant. You must analyze the context and think step-by-step out loud before answering.\n\n"
            "RULES:\n"
            "1. You must ONLY use the information provided in the Context. Do NOT use general knowledge.\n"
            "2. The answer is ALWAYS hidden somewhere in the text. You must search carefully.\n\n"
            f"Context:\n{context or ''}\n\n"
            f"Question: {question}\n\n"
            "Write your step-by-step reasoning inside <scratchpad> tags. "
            "After you are done thinking, conclude your response on a new line with the exact format: 'Final Answer: <exact entity name>'. "
            f"If the answer is completely missing, output 'Final Answer: {_MUSIQUE_NOT_PRESENT}'."
        )
        return prompt, None

    if "fever" in normalized or "fact-verification" in skill_tags:
        prompt = (
            f"Claim: {question}\n\n"
            "Based on the provided context, verify the claim. "
            "Answer strictly with one of these three labels: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
        )
        return prompt, context

    if "hardmath" in normalized or "math" in normalized or "math" in skill_tags:
        prompt = question + "\n\nPlease put your final answer enclosed in \\boxed{}."
        return prompt, context

    if "humaneval" in normalized or "code" in skill_tags:
        return question, context

    prompt = question + "\n\nAnswer the question concisely with just the answer."
    return prompt, context


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
