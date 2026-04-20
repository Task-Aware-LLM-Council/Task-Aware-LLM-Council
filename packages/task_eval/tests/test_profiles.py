from task_eval import FeverProfile, HardMathProfile, MusiqueProfile, QualityProfile, get_dataset_profile


def test_registry_resolves_supported_profile() -> None:
    profile = get_dataset_profile("musique", dataset_name="dummy")
    assert profile.name == "musique"
    assert profile.primary_metric == "token_f1"


def test_musique_profile_scores_qa_answers() -> None:
    profile = MusiqueProfile(dataset_name="dummy")
    case = profile.row_to_case(
        {"id": "1", "question": "Q?", "answer": "Paris", "context": "ctx"},
        0,
    )
    result = profile.score(
        case=case,
        prediction={"response_text": "Final Answer: Paris."},
    )
    assert result.values["exact_match"] == 1.0
    assert result.values["token_f1"] == 1.0


def test_quality_profile_scores_mcq_answers() -> None:
    profile = QualityProfile(dataset_name="dummy")
    case = profile.row_to_case(
        {
            "id": "2",
            "question": "Q?",
            "article": "ctx",
            "options": ["Rome", "Paris", "Berlin", "Tokyo"],
            "answer": 1,
        },
        0,
    )
    result = profile.score(
        case=case,
        prediction={"response_text": "Answer: B"},
    )
    assert result.values["exact_match"] == 1.0


def test_fever_profile_normalizes_labels() -> None:
    profile = FeverProfile(dataset_name="dummy")
    case = profile.row_to_case({"id": "3", "claim": "claim", "label": "SUPPORTED"}, 0)
    result = profile.score(
        case=case,
        prediction={"response_text": "This claim is supported."},
    )
    assert result.values["label_accuracy"] == 1.0


def test_hardmath_profile_uses_numeric_tolerance() -> None:
    profile = HardMathProfile(dataset_name="dummy")
    case = profile.row_to_case({"id": "4", "problem": "2+2", "answer": "4"}, 0)
    result = profile.score(
        case=case,
        prediction={"response_text": "Final Answer: 4.0"},
    )
    assert result.values["numeric_accuracy"] == 1.0
