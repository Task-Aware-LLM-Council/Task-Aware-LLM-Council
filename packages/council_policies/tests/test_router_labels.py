"""
Tests for `router_labels` — the tag → role collapse.

Every dataset loader in `packages/data_prep/src/data_prep/load_*.py`
emits a fixed `skill_tags` list. These tests pin the behavior for
each real row shape, so a tag rename in data_prep fails here instead
of silently relabeling training examples.
"""

from __future__ import annotations

import pytest

from council_policies.p4.router_labels import (
    DEFAULT_FALLBACK_ROLE,
    ROLE_LABELS,
    index_to_role,
    role_from_tags,
    role_to_index,
)


# --------------------------------------------------------------------------- #
# Real row shapes (source: packages/data_prep/src/data_prep/load_*.py)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tags,expected_role",
    [
        (["code"], "math_code"),                      # HumanEvalPlus
        (["math"], "math_code"),                      # HARDMATH
        (["fact-check"], "fact_general"),             # FEVER
        (["retrieval", "multi-hop"], "qa_reasoning"), # MuSiQue
        (["long-context"], "qa_reasoning"),           # QuALITY
    ],
)
def test_known_dataset_rows_map_to_correct_role(tags, expected_role):
    assert role_from_tags(tags) == expected_role


def test_musique_tag_order_insensitive():
    """MuSiQue rows today are always `["retrieval", "multi-hop"]` but
    data_prep could reorder. Both orderings must land on qa_reasoning."""
    assert role_from_tags(["retrieval", "multi-hop"]) == "qa_reasoning"
    assert role_from_tags(["multi-hop", "retrieval"]) == "qa_reasoning"


# --------------------------------------------------------------------------- #
# Edge cases — nothing here should raise
# --------------------------------------------------------------------------- #


def test_empty_tags_return_fallback_role():
    assert role_from_tags([]) == DEFAULT_FALLBACK_ROLE


def test_none_tags_return_fallback_role():
    assert role_from_tags(None) == DEFAULT_FALLBACK_ROLE


def test_unknown_tag_returns_fallback_role():
    """Defensive — if a new loader emits a tag we haven't seen, we
    don't want training to crash on that row. Fallback lets the row
    contribute to the generalist class instead of getting silently
    dropped."""
    assert role_from_tags(["some-brand-new-tag"]) == DEFAULT_FALLBACK_ROLE


# --------------------------------------------------------------------------- #
# Index <-> role round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("role", list(ROLE_LABELS))
def test_role_index_round_trip(role):
    assert index_to_role(role_to_index(role)) == role


def test_role_to_index_rejects_unknown_role():
    """If someone adds a role to the classifier head without updating
    ROLE_LABELS, this blows up at training time instead of emitting
    silently wrong class IDs."""
    with pytest.raises(ValueError, match="not in ROLE_LABELS"):
        role_to_index("not_a_role")


def test_role_labels_is_three_classes():
    """Defense against accidental edit. The classifier head size and
    this list must agree — bumping one without the other corrupts all
    existing artifacts."""
    assert len(ROLE_LABELS) == 3
    assert set(ROLE_LABELS) == {"math_code", "qa_reasoning", "fact_general"}
