from .metrics_analysis import group_by_example, avg_tokens_per_question, avg_calls_per_question

def test_avg_tokens_and_calls_simple_case():
    records = [
        {
            "example_id": "ex1",
            "status": "success",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        {
            "example_id": "ex2",
            "status": "success",
            "usage": {"total_tokens": 20},
        },
    ]

    grouped = group_by_example(records)
    assert set(grouped.keys()) == {"ex1", "ex2"}

    avg_tokens = avg_tokens_per_question(grouped)
    assert avg_tokens == 17.5

    avg_calls = avg_calls_per_question(grouped)
    assert avg_calls == 1.0