"""
Single source of truth for router text featurization.

Both training and inference call `featurize(question, context)`. If
they ever disagree on the format, the classifier silently loses
accuracy — the classic training/serving skew failure. Keeping this
logic in one function in one module is the simplest defense.

The council_policies package owns the inference-time contract, so the
featurizer lives here; `training.train_router` imports it. This is
the same pattern as `DEFAULT_DECOMPOSER_SYSTEM_PROMPT` living in
`decomposer.py` and being consumed by future decomposer fine-tune jobs.
"""

from __future__ import annotations

DEFAULT_CONTEXT_CHAR_CAP = 2048


def featurize(
    question: str,
    context: str = "",
    *,
    context_char_cap: int = DEFAULT_CONTEXT_CHAR_CAP,
) -> str:
    """Serialize (question, context) for the router classifier.

    Format:
      - No context → the question verbatim.
      - With context → `{question}\\n\\nContext:\\n{context[:cap]}`.

    The char cap is an upstream guard before tokenization; the
    tokenizer truncates again to its own max length (512 for
    distilroberta). The cap exists so we don't pay tokenizer time on
    text that will be dropped anyway. Changing the cap requires
    retraining — it's part of the training-time contract, not a
    runtime knob.
    """
    if not context.strip():
        return question
    return f"{question}\n\nContext:\n{context[:context_char_cap]}"
