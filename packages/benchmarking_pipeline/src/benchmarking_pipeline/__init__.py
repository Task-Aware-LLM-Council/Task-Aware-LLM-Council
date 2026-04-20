"""Public API for benchmarking_pipeline.

This package orchestrates normalized benchmark examples through llm_gateway
and persists per-example prediction artifacts plus run metadata.
"""

from benchmarking_pipeline.models import (
    BenchmarkDataset,
    BenchmarkExample,
    BenchmarkPrediction,
    BenchmarkRunConfig,
    BenchmarkRunResult,
)
from benchmarking_pipeline.prompts import build_prompt_request
from benchmarking_pipeline.runner import run_benchmark

__all__ = [
    "BenchmarkDataset",
    "BenchmarkExample",
    "BenchmarkPrediction",
    "BenchmarkRunConfig",
    "BenchmarkRunResult",
    "build_prompt_request",
    "run_benchmark",
]


def main() -> None:
    print("benchmarking_pipeline is a library package. Import run_benchmark(...) to use it.")
