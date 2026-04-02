from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from datasets import load_dataset

from benchmark_runner.models import BenchmarkCase


class DatasetSource(Protocol):
    name: str
    metadata: dict[str, Any]

    def iter_cases(self) -> Iterator[BenchmarkCase]:
        ...


@dataclass(slots=True)
class IterableDatasetSource:
    name: str
    cases: Iterator[BenchmarkCase] | list[BenchmarkCase] | tuple[BenchmarkCase, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def iter_cases(self) -> Iterator[BenchmarkCase]:
        yield from self.cases


@dataclass(slots=True)
class HuggingFaceDatasetSource:
    name: str
    dataset_name: str
    row_mapper: Callable[[dict[str, Any], int], BenchmarkCase]
    split: str = "train"
    config_name: str | None = None
    streaming: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def iter_cases(self) -> Iterator[BenchmarkCase]:
        dataset = load_dataset(
            self.dataset_name,
            self.config_name,
            split=self.split,
            streaming=self.streaming,
        )

        for index, row in enumerate(dataset):
            yield self.row_mapper(dict(row), index)


def chunk_cases(
    source: DatasetSource,
    *,
    batch_size: int,
    max_cases: int | None = None,
) -> Iterator[list[BenchmarkCase]]:
    batch: list[BenchmarkCase] = []
    yielded = 0

    for case in source.iter_cases():
        if max_cases is not None and yielded >= max_cases:
            break

        batch.append(case)
        yielded += 1

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
