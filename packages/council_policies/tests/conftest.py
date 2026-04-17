from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
LLM_GATEWAY_SRC = WORKSPACE_ROOT / "packages" / "llm_gateway" / "src"

TESTS_ROOT = PACKAGE_ROOT / "tests"

for path in (SRC_ROOT, LLM_GATEWAY_SRC, TESTS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
