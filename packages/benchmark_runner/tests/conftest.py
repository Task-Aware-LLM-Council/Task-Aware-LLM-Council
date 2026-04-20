from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
BENCHMARKING_PIPELINE_SRC = WORKSPACE_ROOT / "packages" / "benchmarking_pipeline" / "src"
LLM_GATEWAY_SRC = WORKSPACE_ROOT / "packages" / "llm_gateway" / "src"
TASK_EVAL_SRC = WORKSPACE_ROOT / "packages" / "task_eval" / "src"

for path in (SRC_ROOT, BENCHMARKING_PIPELINE_SRC, LLM_GATEWAY_SRC, TASK_EVAL_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
