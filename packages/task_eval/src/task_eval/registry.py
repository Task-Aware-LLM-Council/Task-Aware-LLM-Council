from __future__ import annotations

from typing import Any

from task_eval.interfaces import DatasetProfile
from task_eval.profiles import (
    FeverProfile,
    HardMathProfile,
    HumanEvalPlusProfile,
    MusiqueProfile,
    QualityProfile,
)


PROFILE_TYPES = {
    "musique": MusiqueProfile,
    "quality": QualityProfile,
    "fever": FeverProfile,
    "hardmath": HardMathProfile,
    "humaneval_plus": HumanEvalPlusProfile,
}


def get_dataset_profile(name: str, **kwargs: Any) -> DatasetProfile:
    normalized = name.strip().lower()
    try:
        profile_type = PROFILE_TYPES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(PROFILE_TYPES))
        raise ValueError(f"Unsupported dataset profile '{name}'. Expected one of: {supported}") from exc
    return profile_type(**kwargs)


def list_dataset_profiles() -> tuple[str, ...]:
    return tuple(sorted(PROFILE_TYPES))
