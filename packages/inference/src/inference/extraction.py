from __future__ import annotations

from task_eval.extraction import (
    extract_code_answer,
    extract_fever_label,
    extract_math_answer,
    extract_qa_answer,
    extract_qa_answer_musique,
)
from task_eval.scoring import (
    label_accuracy,
    math_exact_match,
    token_f1_multi,
)
from task_eval.schemas import TaskTag

_EXTRACTORS = {
    TaskTag.QA_MULTIHOP: extract_qa_answer_musique,
    TaskTag.QA_LONGCTX:  extract_qa_answer,
    TaskTag.FACT_VERIFY: extract_fever_label,
    TaskTag.MATH:        extract_math_answer,
    TaskTag.CODE:        extract_code_answer,
}

_SCORERS = {
    TaskTag.QA_MULTIHOP: lambda pred, refs: token_f1_multi(pred, refs),
    TaskTag.QA_LONGCTX:  lambda pred, refs: token_f1_multi(pred, refs),
    TaskTag.FACT_VERIFY: lambda pred, refs: max((label_accuracy(pred, r) for r in refs), default=0.0),
    TaskTag.MATH:        lambda pred, refs: max((math_exact_match(pred, r) for r in refs), default=0.0),
    TaskTag.CODE:        lambda pred, refs: 0.0,  # pass_at_1 needs test_code + entry_point
}


def extract_answer(raw_text: str, tag: TaskTag) -> str:
    extractor = _EXTRACTORS.get(tag)
    if extractor is None:
        raise ValueError(f"No extractor registered for tag: {tag}")
    return extractor(raw_text)


def score_answer(extracted: str, gold_answers: list[str], tag: TaskTag) -> float:
    scorer = _SCORERS.get(tag)
    if scorer is None:
        raise ValueError(f"No scorer registered for tag: {tag}")
    return scorer(extracted, gold_answers)
