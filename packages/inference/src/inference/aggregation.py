from __future__ import annotations

from collections import Counter
from typing import Sequence

from inference.extraction import extract_answer
from inference.schemas import ModelResponse, TaskTag


def majority_vote(responses: Sequence[ModelResponse], tag: TaskTag) -> str:
    answers = [extract_answer(r.raw_text, tag) for r in responses]
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def aggregate(responses: Sequence[ModelResponse], tag: TaskTag) -> str:
    return majority_vote(responses, tag)
