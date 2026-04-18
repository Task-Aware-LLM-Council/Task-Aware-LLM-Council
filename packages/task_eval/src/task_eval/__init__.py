"""Public API for task_eval.

This package owns dataset-specific case building, response extraction, and
metric calculation for benchmark tasks.
"""

from task_eval.extraction import (
    extract_answer,
    extract_code_answer,
    extract_fever_label,
    extract_math_answer,
    extract_mcq_answer,
    extract_qa_answer,
    extract_qa_answer_musique
)
from task_eval.interfaces import DatasetProfile, MetricCalculator
from task_eval.models import EvaluationCase, MetricResult, PredictionRecord
from task_eval.normalization import normalize_answer, normalize_fever_label, tokenize_normalized
from task_eval.profiles import (
    FeverProfile,
    HardMathProfile,
    HumanEvalPlusProfile,
    MusiqueProfile,
    QualityProfile,
)
from task_eval.registry import get_dataset_profile, list_dataset_profiles
from task_eval.scoring import (
    aggregate_numeric_metrics,
    exact_match,
    exact_match_multi,
    label_accuracy,
    numeric_accuracy,
    pass_at_1,
    token_f1,
    token_f1_multi,
    math_exact_match
)
from .metrics_analysis import (
    find_prediction_files,
    load_predictions,
    group_by_example,
    avg_tokens_per_question,
    avg_calls_per_question,
)

__all__ = [
    "DatasetProfile",
    "MetricCalculator",
    "EvaluationCase",
    "MetricResult",
    "PredictionRecord",
    "MusiqueProfile",
    "QualityProfile",
    "FeverProfile",
    "HardMathProfile",
    "HumanEvalPlusProfile",
    "get_dataset_profile",
    "list_dataset_profiles",
    "normalize_answer",
    "normalize_fever_label",
    "tokenize_normalized",
    "extract_answer",
    "extract_qa_answer",
    "extract_qa_answer_musique",
    "extract_mcq_answer",
    "extract_fever_label",
    "extract_math_answer",
    "extract_code_answer",
    "exact_match",
    "exact_match_multi",
    "token_f1",
    "token_f1_multi",
    "label_accuracy",
    "numeric_accuracy",
    "pass_at_1",
    "aggregate_numeric_metrics",
    "math_exact_match"
]


def main() -> None:
    print("task_eval is a library package. Import dataset profiles or get_dataset_profile(...).")
