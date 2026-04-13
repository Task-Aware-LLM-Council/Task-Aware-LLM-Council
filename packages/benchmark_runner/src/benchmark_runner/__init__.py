"""Public API for benchmark_runner.

This package coordinates dataset streaming, benchmark execution through
benchmarking_pipeline, and generic scoring over model-dataset pairs.
"""

from benchmark_runner.cli import main
from benchmark_runner.config import (
    DATASET_CONFIGS,
    MODEL_POOL,
    build_benchmark_spec,
    default_provider_config,
    get_dataset_configs,
    get_preset_spec,
)
from benchmark_runner.metrics import Metric, MetricResult, NullMetric
from benchmark_runner.models import (
    AggregateMetricRow,
    BenchmarkCase,
    BenchmarkSpec,
    BenchmarkSuiteResult,
    DatasetRunConfig,
    ScoreRecord,
    ScoreSummary,
)
from benchmark_runner.sources import DatasetSource, HuggingFaceDatasetSource, IterableDatasetSource
from benchmark_runner.suite import run_benchmark_suite, run_registered_benchmark_suite
from llm_gateway import (
    VLLMRuntime as ApptainerServerHandle,
    VLLMRuntimeConfig as ApptainerServerConfig,
    managed_local_provider_config,
)

__all__ = [
    "BenchmarkCase",
    "BenchmarkSpec",
    "BenchmarkSuiteResult",
    "DatasetRunConfig",
    "AggregateMetricRow",
    "MODEL_POOL",
    "DATASET_CONFIGS",
    "build_benchmark_spec",
    "default_provider_config",
    "get_dataset_configs",
    "get_preset_spec",
    "ApptainerServerConfig",
    "ApptainerServerHandle",
    "managed_local_provider_config",
    "DatasetSource",
    "HuggingFaceDatasetSource",
    "IterableDatasetSource",
    "Metric",
    "MetricResult",
    "NullMetric",
    "ScoreRecord",
    "ScoreSummary",
    "run_benchmark_suite",
    "run_registered_benchmark_suite",
    "main",
]
