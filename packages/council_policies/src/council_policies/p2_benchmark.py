"""
P2 benchmark runner — mirrors P1's benchmark_runner but uses DatasetCouncilPolicy
instead of a single model.

For each dataset source:
  1. Load cases via source.iter_cases() — same data as P1.
  2. Run P2 council (3 models, blind peer-review rating).
  3. Score best_answer per question with task_eval metrics.
  4. Save scores + summary to JSONL (same layout as benchmark_runner).

Usage (CLI):
    uv run council-p2-bench --provider huggingface --output-root ./results

Usage (Python):
    from council_policies.p2_benchmark import run_p2_benchmark_suite
    result = await run_p2_benchmark_suite(dataset_sources, orchestrator, output_root=Path("results"))
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from model_orchestration import ModelOrchestrator
from task_eval import EvaluationCase, MetricCalculator
from task_eval.scoring import aggregate_numeric_metrics

from council_policies.p2_policy import DatasetCouncilPolicy, P2PolicyResult

logger = logging.getLogger(__name__)


# ── Output models ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class P2ScoreRecord:
    suite_id: str
    dataset_name: str
    example_id: str
    status: str                          # "scored" | "no_ratings" | "skipped" | "metric_error"
    best_model: str | None = None
    best_answer: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    scores_by_role: dict[str, float] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class P2DatasetSummary:
    suite_id: str
    dataset_name: str
    council_winner: str | None         # model that won most questions (by avg score)
    total_examples: int = 0
    scored_examples: int = 0
    failed_examples: int = 0
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    scores_by_role: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class P2BenchmarkResult:
    suite_id: str
    output_dir: Path
    total_examples: int
    scored_examples: int
    failed_examples: int
    skipped_examples: int
    score_files: tuple[Path, ...]
    summary_files: tuple[Path, ...]
    aggregate_summary_path: Path


# ── Storage helpers ────────────────────────────────────────────────────────────

def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_jsonable(asdict(record)), sort_keys=True) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(data), fh, indent=2, sort_keys=True)
        fh.write("\n")


# ── Core runner ────────────────────────────────────────────────────────────────

async def run_p2_benchmark_suite(
    dataset_sources: list[Any],          # DatasetSource — same as P1's sources
    orchestrator: ModelOrchestrator,
    *,
    output_root: Path,
    n_per_dataset: int = 50,
    suite_id: str | None = None,
    metric_resolver: Any | None = None,  # callable(source) -> MetricCalculator
) -> P2BenchmarkResult:
    """
    Run P2 council benchmark over the given dataset sources.

    Args:
        dataset_sources: List of DatasetSource objects (same ones P1 uses).
        orchestrator:    Fully configured ModelOrchestrator with qa/reasoning/general roles.
        output_root:     Directory to write scores, summaries, and aggregate.
        n_per_dataset:   Max questions per dataset.
        suite_id:        Run identifier. Auto-generated if None.
        metric_resolver: Callable(source) -> MetricCalculator. Uses NullMetric if None.
    """
    sid = suite_id or datetime.now(timezone.utc).strftime("p2_suite_%Y%m%dT%H%M%SZ")
    suite_dir = output_root / sid
    (suite_dir / "scores").mkdir(parents=True, exist_ok=True)
    (suite_dir / "summaries").mkdir(parents=True, exist_ok=True)

    score_files: list[Path] = []
    summary_files: list[Path] = []
    total = scored = failed = skipped = 0
    aggregate_rows: list[dict[str, Any]] = []

    policy = DatasetCouncilPolicy(orchestrator, n_per_dataset=n_per_dataset)

    for source in dataset_sources:
        dataset_name: str = source.name
        metric = metric_resolver(source) if metric_resolver else _NullMetric()

        # ── Load cases (same data as P1) ───────────────────────────────────
        cases: list[EvaluationCase] = []
        for case in source.iter_cases():
            cases.append(case)
            if len(cases) >= n_per_dataset:
                break

        if not cases:
            logger.warning("No cases for dataset %r — skipping.", dataset_name)
            continue

        logger.info("P2 benchmark: %d cases from %r", len(cases), dataset_name)

        # ── Run P2 council ─────────────────────────────────────────────────
        result: P2PolicyResult = await policy.run(cases=cases)
        skipped += len(result.skipped_question_ids)

        # ── Score best_answer per question ─────────────────────────────────
        score_file = suite_dir / "scores" / f"{dataset_name}.jsonl"
        score_files.append(score_file)

        pair_records: list[dict[str, Any]] = []

        for qr in result.results:
            total += 1
            example_id = qr.case.example.example_id
            best = qr.best_answer

            if best is None:
                record = P2ScoreRecord(
                    suite_id=sid,
                    dataset_name=dataset_name,
                    example_id=example_id,
                    status="no_ratings",
                    reference=dict(qr.case.reference),
                )
                failed += 1
            else:
                prediction = {"response_text": best.text, "status": "success"}
                try:
                    metric_result = metric.score(
                        case=qr.case,
                        prediction=prediction,
                        dataset_metadata={},
                    )
                    # Build per-role avg scores for this question
                    role_scores = _role_scores_for_question(qr)
                    record = P2ScoreRecord(
                        suite_id=sid,
                        dataset_name=dataset_name,
                        example_id=example_id,
                        status="scored",
                        best_model=best.role,
                        best_answer=best.text,
                        metrics=dict(metric_result.values),
                        reference=dict(qr.case.reference),
                        scores_by_role=role_scores,
                    )
                    scored += 1
                except Exception as exc:
                    logger.warning("Metric failed for %s/%s: %s", dataset_name, example_id, exc)
                    record = P2ScoreRecord(
                        suite_id=sid,
                        dataset_name=dataset_name,
                        example_id=example_id,
                        status="metric_error",
                        best_model=best.role,
                        best_answer=best.text,
                        reference=dict(qr.case.reference),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    failed += 1

            _append_jsonl(score_file, record)
            pair_records.append(asdict(record))

        # ── Dataset summary ────────────────────────────────────────────────
        scored_records = [r for r in pair_records if r["status"] == "scored"]
        agg_metrics = aggregate_numeric_metrics(scored_records) if scored_records else {}

        # council winner for this dataset
        dataset_vote = next(
            (s for s in result.dataset_votes if s.dataset_name == dataset_name), None
        )
        council_winner = dataset_vote.winner if dataset_vote else None
        council_scores = dict(dataset_vote.scores_by_role) if dataset_vote else {}

        summary = P2DatasetSummary(
            suite_id=sid,
            dataset_name=dataset_name,
            council_winner=council_winner,
            total_examples=len(pair_records),
            scored_examples=len(scored_records),
            failed_examples=len(pair_records) - len(scored_records),
            aggregated_metrics=agg_metrics,
            scores_by_role=council_scores,
        )
        summary_file = suite_dir / "summaries" / f"{dataset_name}.json"
        _write_json(summary_file, asdict(summary))
        summary_files.append(summary_file)

        aggregate_rows.append({
            "suite_id": sid,
            "dataset_name": dataset_name,
            "council_winner": council_winner,
            "scores_by_role": council_scores,
            "total_examples": summary.total_examples,
            "scored_examples": summary.scored_examples,
            "aggregated_metrics": agg_metrics,
        })

        logger.info(
            "  %-20s council_winner=%-12s scored=%d/%d",
            dataset_name, council_winner, summary.scored_examples, summary.total_examples,
        )

    aggregate_path = suite_dir / "suite_metrics.json"
    _write_json(aggregate_path, aggregate_rows)

    return P2BenchmarkResult(
        suite_id=sid,
        output_dir=suite_dir,
        total_examples=total,
        scored_examples=scored,
        failed_examples=failed,
        skipped_examples=skipped,
        score_files=tuple(score_files),
        summary_files=tuple(summary_files),
        aggregate_summary_path=aggregate_path,
    )


def _role_scores_for_question(qr: Any) -> dict[str, float]:
    """Average score each role received across all raters for this question."""
    totals: dict[str, list[float]] = {}
    for rr in qr.ratings:
        if rr.failed:
            continue
        for entry in rr.ratings:
            role = rr.label_to_role.get(entry.label)
            if role:
                totals.setdefault(role, []).append(entry.score)
    return {role: sum(s) / len(s) for role, s in totals.items()}


class _NullMetric:
    name = "null_metric"

    def score(self, *, case: Any, prediction: Any, dataset_metadata: Any = None):
        from task_eval.models import MetricResult
        return MetricResult(values={})
