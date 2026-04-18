from __future__ import annotations

from enum import Enum


class TaskTag(str, Enum):
    QA_MULTIHOP = "qa_multihop"
    QA_LONGCTX  = "qa_longctx"
    FACT_VERIFY = "fact_verify"
    MATH        = "math"
    CODE        = "code"

    @classmethod
    def from_dataset_name(cls, dataset: str) -> TaskTag:
        mapping = {
            "musique":        cls.QA_MULTIHOP,
            "quality":        cls.QA_LONGCTX,
            "fever":          cls.FACT_VERIFY,
            "hardmath":       cls.MATH,
            "humaneval_plus": cls.CODE,
        }
        tag = mapping.get(dataset.strip().lower())
        if tag is None:
            raise ValueError(
                f"Unknown dataset: {dataset!r}. Expected one of: {list(mapping)}"
            )
        return tag
