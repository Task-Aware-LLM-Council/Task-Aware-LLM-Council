"""
Router label space + tag → role collapse.

The router is 3-class, not 6-class. `TaskType` stays as eval/audit
metadata; the router only ever outputs one of the three real
specialist roles. See `scratch/plan_p4_learned_router.md` §Label space.

Tag priority is defense against future tag drift: today every
`RouterExample` has tags from a small fixed set (`code`, `math`,
`fact-check`, `retrieval`, `multi-hop`, `long-context`) and the
mapping is unambiguous per-row, but if someone adds a secondary tag
later (e.g. tagging FEVER rows with `reasoning`), the priority table
documents which skill is the target.

Oracle map (from owner, 2026-04-19):

    QuALITY      (long-context)        → Gemma-9B        → qa_reasoning
    MuSiQue      (retrieval,multi-hop) → Gemma-9B        → qa_reasoning
    HumanEvalPlus(code)                → DeepSeek-R1     → math_code
    HardMath     (math)                → DeepSeek-R1     → math_code
    FEVER        (fact-check)          → Qwen-14B        → fact_general
"""

from __future__ import annotations

ROLE_LABELS: tuple[str, ...] = ("math_code", "qa_reasoning", "fact_general")
"""Action space. Order is load-bearing — it pins the integer class
index used by the classifier head. A reorder invalidates any trained
artifact."""

DEFAULT_FALLBACK_ROLE: str = "fact_general"
"""Generalist catches the "don't know" case when no tag matches."""

SKILL_TAG_PRIORITY_TO_ROLE: tuple[tuple[str, str], ...] = (
    ("code", "math_code"),
    ("math", "math_code"),
    ("fact-check", "fact_general"),
    ("multi-hop", "qa_reasoning"),
    ("retrieval", "qa_reasoning"),
    ("long-context", "qa_reasoning"),
)
"""First match wins. Priority order matters only for rows with
multiple tags — today that's just MuSiQue (`retrieval` + `multi-hop`),
where both resolve to the same role, so priority is vacuously
satisfied. The table is kept explicit so future tag additions can
encode their precedence without rewriting label logic."""


def role_from_tags(tags: list[str] | None) -> str:
    """Collapse a tag list to a single role label. Unknown / empty →
    `DEFAULT_FALLBACK_ROLE` (the generalist). Never raises."""
    if not tags:
        return DEFAULT_FALLBACK_ROLE
    for tag, role in SKILL_TAG_PRIORITY_TO_ROLE:
        if tag in tags:
            return role
    return DEFAULT_FALLBACK_ROLE


def role_to_index(role: str) -> int:
    """Stable role → integer class index for the classifier head.
    Raises `ValueError` if `role` isn't in `ROLE_LABELS` — a silent
    unknown-label would corrupt training."""
    try:
        return ROLE_LABELS.index(role)
    except ValueError as exc:
        raise ValueError(
            f"role {role!r} not in ROLE_LABELS {ROLE_LABELS!r}"
        ) from exc


def index_to_role(index: int) -> str:
    """Inverse of `role_to_index`. Raises `IndexError` on out-of-range."""
    return ROLE_LABELS[index]
