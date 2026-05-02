from __future__ import annotations


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
