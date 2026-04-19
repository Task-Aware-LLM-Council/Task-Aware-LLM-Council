"""
Router card — serialized metadata that travels with a trained
classifier artifact.

Two consumers:
  * `training.train_router` writes one per training run.
  * `HFScoreFn` reads it to know the label space, floor, and version
    of the model it just loaded.

The card captures "what the model thinks it is" so we never silently
compare benchmark runs across incompatible router versions. Any time
we retrain, `model_version` changes; eval can filter/group by it.

Schema is versioned via `CARD_SCHEMA_VERSION`. Bump when adding
required fields; bump the loader tolerance at the same time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CARD_SCHEMA_VERSION: int = 1


@dataclass(slots=True)
class RouterCard:
    """Metadata + runtime config for a trained router artifact.

    Parameters
    ----------
    model_version:
        Unique ID for this trained artifact. Surface in
        `CouncilResponse.metadata` so eval can group runs by router
        version. Suggested format: `YYYY-MM-DD-<shortsha>`.
    roles:
        Ordered role label space. Must match
        `router_labels.ROLE_LABELS` at train time; the order pins the
        integer class index of the classifier head.
    floor:
        Default confidence floor for `LearnedRouter`. Pulled from dev-
        set ROC during training; overridable at inference time.
    fallback_role:
        Role used when top-1 probability is below `floor`. Must be in
        `roles` (validated on load).
    base_model:
        HF hub ID of the starting checkpoint. `distilroberta-base` by
        default; captured for reproducibility.
    context_char_cap:
        Featurizer char cap used during training. Mismatched
        inference-time cap silently degrades accuracy — enforced on
        load, not just documented.
    dataset_revision:
        HF dataset commit SHA (or revision string). Lets eval pin
        exactly which router-dataset snapshot the model saw.
    metrics:
        Held-out scores. Free-form dict; common keys: `dev_macro_f1`,
        `dev_confusion` (3×3 list-of-lists), `mini_test_macro_f1`.
    training_config:
        Free-form dict of training hyperparameters (lr, epochs, batch,
        etc.). For audit, not reloaded.
    git_sha:
        Repo SHA at training time. Optional but strongly recommended.
    schema_version:
        Card schema version. Do not set manually — the class sets it.
    """

    model_version: str
    roles: tuple[str, ...]
    floor: float
    fallback_role: str
    base_model: str = "distilroberta-base"
    context_char_cap: int = 2048
    dataset_revision: str = ""
    metrics: dict[str, object] = field(default_factory=dict)
    training_config: dict[str, object] = field(default_factory=dict)
    git_sha: str = ""
    schema_version: int = CARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.fallback_role not in self.roles:
            raise ValueError(
                f"fallback_role {self.fallback_role!r} not in roles "
                f"{self.roles!r}"
            )
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError(f"floor must be in [0, 1], got {self.floor}")

    def save(self, path: str | Path) -> None:
        """Write JSON to `path`. Overwrites. Tuple `roles` is emitted
        as a JSON array and reloaded back to tuple by `load()`."""
        payload = asdict(self)
        payload["roles"] = list(self.roles)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> RouterCard:
        raw = json.loads(Path(path).read_text())
        version = raw.get("schema_version", 0)
        if version > CARD_SCHEMA_VERSION:
            raise ValueError(
                f"RouterCard at {path} has schema_version={version}, "
                f"but this code only understands up to "
                f"{CARD_SCHEMA_VERSION}. Upgrade the library before loading."
            )
        raw["roles"] = tuple(raw["roles"])
        return cls(**raw)
