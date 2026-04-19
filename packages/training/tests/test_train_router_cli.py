"""
Tests for the pure / non-ML surface of `training.train_router`.

The actual `main()` pipeline pulls `torch`/`transformers`/`datasets`
and writes a fine-tuned checkpoint — that belongs in an integration
run, not a unit test. What we CAN pin here without touching the ML
stack:

  * Argparse surface: required flags, defaults, type coercion.
  * `_model_version()` format (date-shape + base-model name).
  * Row → example conversion against real-shape rows (question,
    context, skill_tags), matched against
    `council_policies.router_labels.role_from_tags`.

Together these catch the most common regression modes — CLI drift
and label-map drift — without needing the training stack.
"""

from __future__ import annotations

import pytest

from training.train_router import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASET,
    _model_version,
    _rows_to_examples,
    parse_args,
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_parse_args_requires_out():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_defaults():
    args = parse_args(["--out", "/tmp/router"])

    assert str(args.out) == "/tmp/router"
    assert args.dataset == DEFAULT_DATASET
    assert args.base_model == DEFAULT_BASE_MODEL
    assert args.epochs == 5
    assert args.lr == pytest.approx(2e-5)
    assert args.batch_size == 16
    assert args.max_length == 512
    assert args.dev_eval_frac == pytest.approx(0.2)
    assert args.seed == 42
    assert args.floor == pytest.approx(0.3)


def test_parse_args_overrides():
    args = parse_args(
        [
            "--out", "/tmp/router",
            "--epochs", "3",
            "--lr", "1e-4",
            "--batch-size", "8",
            "--dataset-revision", "abc123",
            "--seed", "7",
            "--floor", "0.5",
        ]
    )

    assert args.epochs == 3
    assert args.lr == pytest.approx(1e-4)
    assert args.batch_size == 8
    assert args.dataset_revision == "abc123"
    assert args.seed == 7
    assert args.floor == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Model version string
# --------------------------------------------------------------------------- #


def test_model_version_contains_base_model_and_date():
    v = _model_version("distilroberta-base")
    # YYYY-MM-DD at the start; base name appears; git sha or 'nosha'
    # tail. We don't pin the exact sha because it depends on repo state.
    assert v.startswith(("2024-", "2025-", "2026-", "2027-"))
    assert "distilroberta-base" in v


def test_model_version_strips_hf_org_prefix():
    v = _model_version("FacebookAI/roberta-base")
    assert "roberta-base" in v
    assert "FacebookAI" not in v


# --------------------------------------------------------------------------- #
# Row → example — the label-map contract surface
# --------------------------------------------------------------------------- #


def test_rows_to_examples_maps_real_dataset_shapes():
    """Pin the full (featurize + label) behavior against one row per
    real source dataset. A silent tag rename in data_prep, or a
    featurizer drift, blows up here."""
    from council_policies.router_featurize import featurize
    from council_policies.router_labels import role_from_tags, role_to_index

    rows = [
        {"question": "Write a fn", "context": "", "skill_tags": ["code"]},
        {"question": "2+2", "context": "", "skill_tags": ["math"]},
        {"question": "True?", "context": "", "skill_tags": ["fact-check"]},
        {
            "question": "Multi-hop Q",
            "context": "",
            "skill_tags": ["retrieval", "multi-hop"],
        },
        {
            "question": "Read this",
            "context": "long passage",
            "skill_tags": ["long-context"],
        },
    ]
    texts, labels = _rows_to_examples(
        rows, featurize, role_from_tags, role_to_index
    )

    assert texts[0] == "Write a fn"
    assert texts[4] == "Read this\n\nContext:\nlong passage"
    assert [role_to_index(r) for r in
            ("math_code", "math_code", "fact_general",
             "qa_reasoning", "qa_reasoning")] == labels


def test_rows_to_examples_handles_missing_fields():
    """Real rows sometimes omit `context` or have `skill_tags=None`
    after HF roundtrip — don't crash. Unknown label → fallback role,
    not dropped."""
    from council_policies.router_featurize import featurize
    from council_policies.router_labels import (
        DEFAULT_FALLBACK_ROLE,
        role_from_tags,
        role_to_index,
    )

    rows = [
        {"question": "Q1", "context": None, "skill_tags": None},
        {"question": "Q2"},  # both fields missing
    ]

    texts, labels = _rows_to_examples(
        rows, featurize, role_from_tags, role_to_index
    )

    assert texts == ["Q1", "Q2"]
    assert labels == [role_to_index(DEFAULT_FALLBACK_ROLE)] * 2
