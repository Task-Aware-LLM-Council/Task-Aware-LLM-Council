"""
HuggingFace-backed `ScoreFn` adapter.

Loads a fine-tuned classifier (saved by `training.train_router`) and
satisfies the `ScoreFn` protocol consumed by `LearnedRouter`.

Heavy deps (`torch`, `transformers`) are imported lazily inside
`_load()` so `import council_policies` stays light — only callers
that actually instantiate `HFScoreFn` pay for the ML stack. That
matters for CI runs, benchmark drivers that don't use P4, and every
other code path that touches this package.

Typical wiring:

    card = RouterCard.load("artifacts/router_model/router_card.json")
    score_fn = HFScoreFn(
        model_dir="artifacts/router_model",
        roles=card.roles,
    )
    router = LearnedRouter(
        score_fn=score_fn,
        fallback_role=card.fallback_role,
        floor=card.floor,
        context_char_cap=card.context_char_cap,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HFScoreFn:
    """Lazy wrapper around `transformers.AutoModelForSequenceClassification`.

    Parameters
    ----------
    model_dir:
        Path to the saved artifact — a directory containing the
        tokenizer files + model weights + config (as written by
        `save_pretrained`). Not loaded until the first `__call__`.
    roles:
        Ordered role labels, matching the integer class indices of
        the classifier head. Read from `RouterCard.roles`. The HF
        model's `config.id2label` is NOT trusted — callers own the
        label mapping, not the checkpoint.
    device:
        `"cpu"` by default. The online router path is CPU-only per
        the P4 plan; one forward pass is ~50 ms, no GPU needed.
        Override to `"cuda"` for batch inference jobs.
    max_length:
        Tokenizer truncation. Defaults to 512 (distilroberta's
        native max). Values above the model's native max are
        clamped by the tokenizer, so this is effectively an upper
        bound.
    """

    def __init__(
        self,
        *,
        model_dir: str | Path,
        roles: tuple[str, ...],
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        if not roles:
            raise ValueError("roles must be non-empty")
        self.model_dir = Path(model_dir)
        self.roles = tuple(roles)
        self.device = device
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        """First-call loader. Imports torch/transformers inside the
        function so module import cost stays near zero."""
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        logger.info("HFScoreFn: loading router model from %s", self.model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )
        model.eval()
        model.to(self.device)
        self._model = model

        head_size = int(model.config.num_labels)
        if head_size != len(self.roles):
            raise ValueError(
                f"HFScoreFn: model head has {head_size} classes but "
                f"roles has {len(self.roles)} labels "
                f"({self.roles!r}). Training/inference label-space drift."
            )
        # Eager-import torch so __call__ can reuse it without
        # re-importing on every call.
        self._torch = torch

    def __call__(self, text: str) -> dict[str, float]:
        if self._model is None:
            self._load()
        assert self._tokenizer is not None and self._model is not None

        enc = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self._torch.no_grad():
            logits = self._model(**enc).logits
        probs = self._torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return dict(zip(self.roles, probs, strict=True))
