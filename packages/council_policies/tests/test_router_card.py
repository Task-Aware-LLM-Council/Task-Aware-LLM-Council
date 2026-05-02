"""
Tests for `RouterCard` — validation + JSON round-trip.

The card is the one handoff point between training and inference.
If save/load drifts, an artifact trained on one git SHA deserializes
wrong on another and the router silently mis-labels. The tests here
pin the disk format.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from council_policies.p4.router_card import CARD_SCHEMA_VERSION, RouterCard


def _minimal_card(**overrides) -> RouterCard:
    defaults = dict(
        model_version="2026-04-19-testsha",
        roles=("math_code", "qa_reasoning", "fact_general"),
        floor=0.3,
        fallback_role="fact_general",
    )
    defaults.update(overrides)
    return RouterCard(**defaults)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_rejects_fallback_role_not_in_roles():
    with pytest.raises(ValueError, match="fallback_role"):
        _minimal_card(fallback_role="something_else")


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_rejects_floor_outside_unit_interval(bad):
    with pytest.raises(ValueError, match="floor"):
        _minimal_card(floor=bad)


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


def test_save_load_round_trip_preserves_fields(tmp_path: Path):
    original = _minimal_card(
        metrics={"dev_macro_f1": 0.87, "confusion": [[5, 0, 0]]},
        training_config={"epochs": 3, "lr": 2e-5},
        dataset_revision="abc1234",
        git_sha="deadbeef",
    )
    path = tmp_path / "router_card.json"
    original.save(path)

    loaded = RouterCard.load(path)

    assert loaded.model_version == original.model_version
    assert loaded.roles == original.roles
    assert isinstance(loaded.roles, tuple)  # list-from-JSON → tuple restored
    assert loaded.floor == original.floor
    assert loaded.fallback_role == original.fallback_role
    assert loaded.metrics == original.metrics
    assert loaded.training_config == original.training_config
    assert loaded.dataset_revision == original.dataset_revision
    assert loaded.git_sha == original.git_sha
    assert loaded.schema_version == CARD_SCHEMA_VERSION


def test_load_rejects_future_schema_version(tmp_path: Path):
    """An artifact saved by a newer library version should refuse to
    load rather than quietly lose fields. Forward-incompatibility here
    is the safe default — benchmarks that compare runs across weeks
    depend on the router meaning the same thing."""
    import json

    path = tmp_path / "future_card.json"
    path.write_text(
        json.dumps(
            {
                "model_version": "future",
                "roles": ["a", "b", "c"],
                "floor": 0.3,
                "fallback_role": "a",
                "base_model": "x",
                "context_char_cap": 2048,
                "dataset_revision": "",
                "metrics": {},
                "training_config": {},
                "git_sha": "",
                "schema_version": CARD_SCHEMA_VERSION + 1,
            }
        )
    )
    with pytest.raises(ValueError, match="schema_version"):
        RouterCard.load(path)


def test_saved_json_is_indented_and_sorted(tmp_path: Path):
    """Card JSON is committed as an audit trail; stable key order keeps
    diffs readable."""
    path = tmp_path / "card.json"
    _minimal_card().save(path)
    raw = path.read_text()
    assert raw.startswith("{")
    assert "\n  " in raw  # indent=2
    # Keys sorted alphabetically: `base_model` appears before `floor`.
    assert raw.index('"base_model"') < raw.index('"floor"')
