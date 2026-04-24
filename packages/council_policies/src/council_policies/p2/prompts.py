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
    """Ask the synthesizer to perfectly format the winning answer based on the dataset."""
    normalized = (source_dataset or "").strip().lower()
    others = "\n\n".join(f"- {a}" for a in other_answers)
    context_block = f"Context:\n{context}\n\n" if context else ""

    if "musique" in normalized:
        return (
            "You are a strict output formatter. Your job is to extract the final answer from the primary response.\n\n"
            "STRICT RULES:\n"
            "1. Output the answer on a new line in the exact format: Final Answer: <answer>\n"
            "2. If the primary answer concludes the information is missing, output exactly: Final Answer: NOT PRESENT IN CONTEXT\n"
            "3. You must DISCARD all reasoning, steps, and explanations.\n\n"
            f"Question: {question}\n\n"
            f"{context_block}"
            f"Primary answer (voted best):\n{winning_answer}\n\n"
            f"Other candidate answers:\n{others}\n\n"
            "Normalized final answer:"
        )

    if "fever" in normalized:
        return (
            "You are a strict output formatter. Your job is to extract the final label from the primary response.\n\n"
            "STRICT RULES:\n"
            "1. Output exactly one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO.\n"
            "2. Do NOT include any explanations, reasoning, or introductory text.\n\n"
            f"Claim: {question}\n\n"
            f"{context_block}"
            f"Primary answer (voted best):\n{winning_answer}\n\n"
            f"Other candidate answers:\n{others}\n\n"
            "Normalized final answer:"
        )

    if "hardmath" in normalized or "math" in normalized:
        return (
            "You are a strict output formatter. Your job is to extract the final mathematical answer from the primary response.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the final answer enclosed in \\boxed{}.\n"
            "2. Do NOT output the step-by-step derivation, prose, or commentary.\n\n"
            f"Question: {question}\n\n"
            f"Primary answer (voted best):\n{winning_answer}\n\n"
            f"Other candidate answers:\n{others}\n\n"
            "Normalized final answer:"
        )

    if "humaneval" in normalized or "code" in normalized:
        return (
            "You are a strict output formatter. Your job is to extract the final executable code from the primary response.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY executable Python code.\n"
            "2. Do NOT include any explanations, prose, or introductory text (e.g. discard 'Here is the code:').\n\n"
            f"Question:\n{question}\n\n"
            f"Primary answer (voted best):\n{winning_answer}\n\n"
            f"Other candidate answers:\n{others}\n\n"
            "Normalized final answer:\n"
        )

    # Fallback for Quality / general QA
    return (
        "You are a strict output formatter. Your job is to extract the final concise answer from the primary response.\n\n"
        "STRICT RULES:\n"
        "1. Output ONLY a concise direct answer with no explanation.\n"
        "2. DISCARD all reasoning and commentary.\n\n"
        f"Question: {question}\n\n"
        f"{context_block}"
        f"Primary answer (voted best):\n{winning_answer}\n\n"
        f"Other candidate answers:\n{others}\n\n"
        "Normalized final answer:"
    )
