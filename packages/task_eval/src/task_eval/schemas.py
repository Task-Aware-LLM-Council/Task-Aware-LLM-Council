from __future__ import annotations

from enum import Enum


class TaskTag(str, Enum):
    QA_MULTIHOP = "qa_multihop"
    QA_LONGCTX  = "qa_longctx"
    FACT_VERIFY = "fact_verify"
    MATH        = "math"
    CODE        = "code"
