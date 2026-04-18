from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from task_eval.interfaces import DatasetProfile
from task_eval.profiles import (
    FeverProfile,
    HardMathProfile,
    HumanEvalPlusProfile,
    MusiqueProfile,
    QualityProfile,
)
from task_eval.schemas import TaskTag

logger = logging.getLogger(__name__)


# ── Dataset profile registry ───────────────────────────────────────────────────

PROFILE_TYPES = {
    "musique":       MusiqueProfile,
    "quality":       QualityProfile,
    "fever":         FeverProfile,
    "hardmath":      HardMathProfile,
    "humaneval_plus": HumanEvalPlusProfile,
}


def get_dataset_profile(name: str, **kwargs: Any) -> DatasetProfile:
    normalized = name.strip().lower()
    try:
        profile_type = PROFILE_TYPES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(PROFILE_TYPES))
        raise ValueError(
            f"Unsupported dataset profile '{name}'. Expected one of: {supported}"
        ) from exc
    return profile_type(**kwargs)


def list_dataset_profiles() -> tuple[str, ...]:
    return tuple(sorted(PROFILE_TYPES))


# ── Specialist registry ────────────────────────────────────────────────────────
# File lives at:  packages/task_eval/src/task_eval/registry.py
#                           [0]      [1] [2]   [3]
# parents[4] = repo root
_REPO_ROOT    = Path(__file__).resolve().parents[4]
_DEFAULT_PATH = Path(os.getenv(
    "SPECIALISTS_PATH",
    str(_REPO_ROOT / "results" / "specialists.json"),
))

_FALLBACK_SPECIALISTS: dict[str, list[str] | str] = {
    TaskTag.QA_MULTIHOP.value: ["internlm/internlm2.5-7b-chat-1m"],
    TaskTag.QA_LONGCTX.value:  ["Qwen/Qwen2.5-14B-Instruct"],
    TaskTag.FACT_VERIFY.value: ["Qwen/Qwen2.5-7B-Instruct"],
    TaskTag.MATH.value:        ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
    TaskTag.CODE.value:        ["deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"],
    "_best_overall": "Qwen/Qwen2.5-14B-Instruct",
    "_full_pool": [
        "01-ai/Yi-1.5-9B-Chat-16K",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "google/gemma-2-9b-it",
        "internlm/internlm2.5-7b-chat-1m",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "NousResearch/Hermes-3-Llama-3.1-8B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "THUDM/glm-4-9b-chat",
    ],
}


class SpecialistRegistry:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _DEFAULT_PATH
        self._data: dict[str, list[str] | str] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                self._data = json.load(f)
            logger.info("Loaded specialists from %s", self._path)
        else:
            logger.warning(
                "specialists.json not found at %s — using fallback defaults.\n"
                "Run the specialist-selection script first:\n"
                "  uv run -m data_prep.select_specialists",
                self._path,
            )
            self._data = _FALLBACK_SPECIALISTS

    def specialists_for(self, tag: TaskTag) -> list[str]:
        models = self._data.get(tag.value, [])
        if isinstance(models, str):
            models = [models]
        if not models:
            logger.warning("No specialists for tag=%s; falling back to full pool", tag)
            return self.full_pool()
        return list(models)

    def full_pool(self) -> list[str]:
        pool = self._data.get("_full_pool", [])
        if not pool:
            pool = [
                m
                for k, v in self._data.items()
                if not k.startswith("_")
                for m in (v if isinstance(v, list) else [v])
            ]
        return list(pool)

    def best_overall(self) -> str:
        best = self._data.get("_best_overall")
        if best and isinstance(best, str):
            return best
        pool = self.full_pool()
        if pool:
            return pool[0]
        raise RuntimeError("Specialist registry is empty")

    def reload(self) -> None:
        self._load()


specialist_registry = SpecialistRegistry()
