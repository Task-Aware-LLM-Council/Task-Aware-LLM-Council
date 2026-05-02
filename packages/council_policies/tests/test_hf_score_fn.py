"""
Tests for `HFScoreFn` — the HuggingFace-backed `ScoreFn` adapter.

The real `transformers`/`torch` stack is optional (under the `p4-ml`
extra), so tests stub both via `sys.modules` rather than pulling them
in. That keeps the unit test suite lean AND exercises the lazy-load
contract — if someone regresses `HFScoreFn.__init__` into doing
eager imports, these tests will fail before any stub is installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from council_policies.p4.hf_score_fn import HFScoreFn


class _FakeTensor:
    """Minimal tensor stand-in: supports `.to()` (returns self) and
    the `.squeeze(0).tolist()` chain used in `HFScoreFn.__call__`."""

    def __init__(self, data):
        self._data = data

    def to(self, _device):
        return self

    def squeeze(self, _dim):
        return _FakeTensor(self._data[0])

    def tolist(self):
        return self._data


class _FakeModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    def __init__(self, logits: list[list[float]], num_labels: int):
        self._logits = _FakeTensor(logits)
        self.config = types.SimpleNamespace(num_labels=num_labels)

    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, **_kwargs):
        return _FakeModelOutput(self._logits)


class _FakeTokenizer:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def __call__(self, text, *, truncation, max_length, return_tensors):
        self.calls.append((text, max_length))
        # HFScoreFn expects a mapping of tensors; one entry is enough.
        return {"input_ids": _FakeTensor([[1, 2, 3]])}


def _install_fake_stack(
    monkeypatch,
    *,
    logits: list[list[float]],
    num_labels: int,
) -> _FakeTokenizer:
    """Register fake `torch` and `transformers` in sys.modules. Returns
    the tokenizer instance so tests can assert what it saw."""
    fake_torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]

    def _softmax(tensor, *, dim):  # noqa: ARG001
        # Stub: skip the real softmax — tests pass pre-softmaxed logits
        # as "probabilities" so they can assert the exact mapping
        # that reaches `LearnedRouter`.
        return tensor

    fake_torch.softmax = _softmax  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_tx = types.ModuleType("transformers")
    fake_tok = _FakeTokenizer()
    fake_model = _FakeModel(logits, num_labels)

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(_path):
            return fake_tok

    class _AutoModel:
        @staticmethod
        def from_pretrained(_path):
            return fake_model

    fake_tx.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    fake_tx.AutoModelForSequenceClassification = _AutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_tx)

    return fake_tok


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_init_is_lazy(tmp_path: Path, monkeypatch):
    """`__init__` must not import torch/transformers. We install the
    fake stack AFTER construction; if init were eager it would crash
    before getting here."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)

    fn = HFScoreFn(
        model_dir=tmp_path,
        roles=("math_code", "qa_reasoning", "fact_general"),
    )

    assert fn._model is None
    assert fn._tokenizer is None


def test_rejects_empty_roles(tmp_path: Path):
    with pytest.raises(ValueError, match="roles"):
        HFScoreFn(model_dir=tmp_path, roles=())


# --------------------------------------------------------------------------- #
# Forward pass
# --------------------------------------------------------------------------- #


def test_call_returns_role_probability_dict(tmp_path: Path, monkeypatch):
    roles = ("math_code", "qa_reasoning", "fact_general")
    _install_fake_stack(
        monkeypatch,
        logits=[[0.7, 0.2, 0.1]],
        num_labels=3,
    )
    fn = HFScoreFn(model_dir=tmp_path, roles=roles)

    scores = fn("Solve 2+2")

    assert set(scores.keys()) == set(roles)
    assert scores["math_code"] == pytest.approx(0.7)
    assert scores["qa_reasoning"] == pytest.approx(0.2)
    assert scores["fact_general"] == pytest.approx(0.1)


def test_call_loads_model_on_first_call_only(tmp_path: Path, monkeypatch):
    """Second invocation must reuse the already-loaded model. Reloading
    on every call would be a ~200ms-per-subtask regression in prod."""
    tok = _install_fake_stack(
        monkeypatch,
        logits=[[0.9, 0.05, 0.05]],
        num_labels=3,
    )
    fn = HFScoreFn(
        model_dir=tmp_path,
        roles=("math_code", "qa_reasoning", "fact_general"),
    )

    fn("first call")
    loaded_once = fn._model
    fn("second call")

    assert fn._model is loaded_once
    # Tokenizer was called twice (once per `fn(...)`), model loaded once.
    assert len(tok.calls) == 2


def test_raises_when_head_size_mismatches_roles(tmp_path: Path, monkeypatch):
    """Loading a checkpoint trained on N classes and configuring the
    adapter for M ≠ N is a training/inference drift bug. Fail hard —
    don't silently emit one-probability-short dicts."""
    _install_fake_stack(
        monkeypatch,
        logits=[[0.5, 0.5]],
        num_labels=2,
    )
    fn = HFScoreFn(
        model_dir=tmp_path,
        roles=("math_code", "qa_reasoning", "fact_general"),
    )

    with pytest.raises(ValueError, match="head"):
        fn("anything")


def test_tokenizer_receives_truncation_at_max_length(
    tmp_path: Path, monkeypatch
):
    tok = _install_fake_stack(
        monkeypatch,
        logits=[[1.0, 0.0, 0.0]],
        num_labels=3,
    )
    fn = HFScoreFn(
        model_dir=tmp_path,
        roles=("math_code", "qa_reasoning", "fact_general"),
        max_length=256,
    )

    fn("hello world")

    assert tok.calls == [("hello world", 256)]
