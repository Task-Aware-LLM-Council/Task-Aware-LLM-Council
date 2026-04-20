from benchmark_runner import BenchmarkCase, IterableDatasetSource
from benchmark_runner.sources import chunk_cases
from benchmarking_pipeline import BenchmarkExample


def test_chunk_cases_limits_materialized_batch_size() -> None:
    source = IterableDatasetSource(
        name="demo",
        cases=[
            BenchmarkCase(
                example=BenchmarkExample(
                    example_id=str(index),
                    dataset_name="demo",
                    question=f"Question {index}",
                ),
                reference={"answer": str(index)},
            )
            for index in range(5)
        ],
    )

    chunks = list(chunk_cases(source, batch_size=2))

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert chunks[0][0].example.example_id == "0"
    assert chunks[-1][0].example.example_id == "4"
