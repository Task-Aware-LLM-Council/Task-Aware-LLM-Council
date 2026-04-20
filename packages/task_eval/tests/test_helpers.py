from task_eval import (
    exact_match,
    extract_code_answer,
    extract_fever_label,
    extract_math_answer,
    extract_mcq_answer,
    extract_qa_answer,
    label_accuracy,
    normalize_answer,
    numeric_accuracy,
    token_f1,
)


def test_normalize_answer_uses_squad_style_cleanup() -> None:
    assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"


def test_extractors_cover_dataset_specific_patterns() -> None:
    assert extract_qa_answer("Final Answer: Paris.") == "Paris"
    assert extract_mcq_answer("The answer is (B).") == "B"
    assert extract_fever_label("Not enough information.") == "NEI"
    assert extract_math_answer("After solving, the answer is 42.") == "42"
    assert extract_code_answer("```python\ndef add(a, b):\n    return a + b\n```").startswith("def add")


def test_scoring_helpers_return_expected_values() -> None:
    assert exact_match("Paris", "paris") == 1.0
    assert token_f1("Paris city", "Paris") > 0.0
    assert label_accuracy("supports", "SUPPORTED") == 1.0
    assert numeric_accuracy("3.14159", "3.14", rel_tol=0.01) == 1.0
