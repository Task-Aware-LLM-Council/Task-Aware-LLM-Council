from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from inference.schemas import TaskTag

logger = logging.getLogger(__name__)

# File lives at:  packages/inference/src/inference/registry.py
#                           [0]     [1] [2]   [3]
# parents[4] = repo root
_THIS_FILE   = Path(__file__).resolve()
_REPO_ROOT   = _THIS_FILE.parents[4]
_RESULTS_DIR = _REPO_ROOT / "results"

_DEFAULT_PATH = Path(os.getenv(
    "SPECIALISTS_PATH",
    str(_RESULTS_DIR / "specialists.json"),
))

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
            raise FileNotFoundError(
                f"specialists.json not found at {self._path}.\n"
                "Run the specialist-selection script first:\n"
                "  uv run -m data_prep.select_specialists"
            )

    def specialists_for(self, tag: TaskTag) -> list[str]:
        models = self._data.get(tag.value, [])
        if isinstance(models, str):
            models = [models]
        if not models:
            raise KeyError(f"No specialists registered for tag={tag!r} in {self._path}")
        return list(models)

    def full_pool(self) -> list[str]:
        pool = self._data.get("_full_pool", [])
        if not pool:
            raise KeyError(f"'_full_pool' key missing in {self._path}")
        return list(pool)

    def best_overall(self) -> str:
        best = self._data.get("_best_overall")
        if best and isinstance(best, str):
            return best
        raise KeyError(f"'_best_overall' key missing in {self._path}")

    def reload(self) -> None:
        self._load()


registry = SpecialistRegistry()
