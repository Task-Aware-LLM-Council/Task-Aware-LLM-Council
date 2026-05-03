"""
P3 Rule-Based Routing Policy — keyword-based specialist dispatch.

Flow per question
-----------------
1. Classify the task type from the question text using keyword regexes
   (classify_task).
2. Map the task type to the matching specialist role via TASK_TO_ROLE.
3. Dispatch to that specialist's model via the orchestrator.
4. If the specialist role is not registered in the orchestrator, fall back
   gracefully to ``fallback_role`` rather than crashing.
5. Return the routing result wrapped in a P3QuestionResult.

Metrics (dataset-level)
-----------------------
compute_routing_summary() aggregates per-question results into
RoutingSummary objects — one per dataset — reporting:
  • how many questions were routed to each role
  • task-type distribution
  • average response latency per role

The aggregator (separate module if needed) can use P3PolicyResult together
with task_eval scoring to evaluate which routing strategy produced the
best answers across datasets.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator, OrchestratorResponse

from council_policies.models import TASK_TO_ROLE, TaskType

if TYPE_CHECKING:
    from task_eval.interfaces import DatasetProfile
    from task_eval.models import EvaluationCase

logger = logging.getLogger(__name__)

_DEFAULT_COUNCIL_ROLES = ("qa", "reasoning", "general")


# ── Regex classifiers ─────────────────────────────────────────────────────────

_MATH_RE = re.compile(
    r"\b(solve|calculat|comput|integrat|differentiat|derivativ|equation|inequalit|"
    r"theorem|proof|algebra|geometr|trigonometr|probabilit|statistic|matric|vector|"
    r"polynomial|logarithm|exponential|fraction|percent|modulo|combinat|permut)\w*\b",
    re.IGNORECASE,
)

_CODE_RE = re.compile(
    r"\b(code|function|implement|algorithm|debug|program|class|method|variable|"
    r"python|javascript|typescript|java|c\+\+|golang|rust|sql|bash|script|api|"
    r"loop|recursion|sort|parse|regex|test|unittest|bug|fix|refactor|deploy)\w*\b",
    re.IGNORECASE,
)

_REASONING_RE = re.compile(
    r"\b(reason|analys|infer|deduc|logic|argument|cause|effect|explain why|"
    r"compare|contrast|evaluat|assess|critic|implicat|consequenc|hypothes|"
    r"evidence|conclus|step.by.step|think through)\w*\b",
    re.IGNORECASE,
)

_FEVER_RE = re.compile(
    r"\b(true or false|fact.check|verif|claim|supports|refutes|evidence|"
    r"fact|accurate|correct|wrong|misinform|debunk)\w*\b",
    re.IGNORECASE,
)

# Dataset-name → task type mapping.  Checked before keyword matching so that
# FEVER/HardMath/HumanEval+ rows are never misclassified as qa.
_DATASET_TO_TASK: dict[str, TaskType] = {
    "musique": TaskType.QA,
    "quality": TaskType.QA,
    "fever": TaskType.FEVER,
    "hardmath": TaskType.MATH,
    "humaneval": TaskType.CODE,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all_profiles() -> list[DatasetProfile]:
    """Return one instance of every available dataset profile."""
    from task_eval.profiles import (
        FeverProfile,
        HardMathProfile,
        HumanEvalPlusProfile,
        MusiqueProfile,
        QualityProfile,
    )
    return [
        MusiqueProfile(),
        QualityProfile(),
        FeverProfile(),
        HardMathProfile(),
        HumanEvalPlusProfile(),
    ]


def classify_task(prompt: str) -> TaskType:
    """Rule-based task classifier using keyword pattern matching."""
    text = prompt.lower()

    scores: dict[TaskType, int] = {
        TaskType.MATH: len(_MATH_RE.findall(text)),
        TaskType.CODE: len(_CODE_RE.findall(text)),
        TaskType.REASONING: len(_REASONING_RE.findall(text)),
        TaskType.FEVER: len(_FEVER_RE.findall(text)),
    }

    best_type, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score > 0:
        return best_type

    return TaskType.QA


# ── Output models ─────────────────────────────────────────────────────────────

@dataclass
class P3QuestionResult:
    """Routing outcome for a single question."""

    case: EvaluationCase
    task_type: TaskType
    routed_role: str
    response: OrchestratorResponse | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return bool(self.error) or self.response is None

    @property
    def latency_ms(self) -> float | None:
        if self.response is None:
            return None
        return self.response.latency_ms


@dataclass
class RoutingSummary:
    """Per-dataset aggregate: routing distribution and average latency."""

    dataset_name: str
    question_count: int
    role_counts: dict[str, int]        # role → number of questions routed to it
    task_type_counts: dict[str, int]   # task_type value → count
    avg_latency_ms: float | None       # across all successful questions in this dataset


@dataclass
class P3PolicyResult:
    results: list[P3QuestionResult] = field(default_factory=list)
    skipped_question_ids: list[str] = field(default_factory=list)
    routing_summaries: list[RoutingSummary] = field(default_factory=list)


# ── Dataset-level routing aggregation ─────────────────────────────────────────

def compute_routing_summary(results: list[P3QuestionResult]) -> list[RoutingSummary]:
    """
    Aggregate per-question routing results into per-dataset RoutingSummary objects.

    For each dataset, reports how often each specialist role was chosen
    and the distribution of classified task types.
    """
    totals: dict[str, dict] = {}

    for qr in results:
        dataset = qr.case.example.dataset_name
        if dataset not in totals:
            totals[dataset] = {
                "role_counts": {},
                "task_type_counts": {},
                "latencies": [],
            }
        bucket = totals[dataset]

        role = qr.routed_role
        bucket["role_counts"][role] = bucket["role_counts"].get(role, 0) + 1

        tt = qr.task_type.value
        bucket["task_type_counts"][tt] = bucket["task_type_counts"].get(tt, 0) + 1

        if qr.latency_ms is not None:
            bucket["latencies"].append(qr.latency_ms)

    summaries: list[RoutingSummary] = []
    for dataset, bucket in totals.items():
        latencies = bucket["latencies"]
        avg_lat = sum(latencies) / len(latencies) if latencies else None
        question_count = sum(bucket["role_counts"].values())
        summaries.append(RoutingSummary(
            dataset_name=dataset,
            question_count=question_count,
            role_counts=dict(bucket["role_counts"]),
            task_type_counts=dict(bucket["task_type_counts"]),
            avg_latency_ms=avg_lat,
        ))

    summaries.sort(key=lambda s: s.dataset_name)
    return summaries


# ── P3 Policy ─────────────────────────────────────────────────────────────────

class RuleBasedRoutingPolicy:
    """
    P3: Route each question to the specialist model best suited for the task.

    Usage::

        orchestrator = ModelOrchestrator(config)
        async with orchestrator:
            policy = RuleBasedRoutingPolicy(orchestrator)
            result = await policy.run()   # samples all datasets automatically

    Parameters
    ----------
    orchestrator:
        A fully-configured ModelOrchestrator. Must have at least
        ``fallback_role`` registered — validated at construction time.
    fallback_role:
        Role used when the classified role is not registered.
        Defaults to ``"general"``.
    n_per_dataset:
        Max questions sampled per dataset profile. Defaults to 5.
    max_concurrent_questions:
        Semaphore limit for concurrent dispatch calls. Defaults to 4.
    seed:
        Random seed for reproducible sampling order.
    """

    def __init__(
        self,
        orchestrator: ModelOrchestrator,
        *,
        fallback_role: str = "general",
        n_per_dataset: int = 5,
        max_concurrent_questions: int = 4,
        seed: int = 42,
    ) -> None:
        # Validate the fallback role is registered — fail fast rather than
        # silently at inference time on the first miss.
        try:
            orchestrator.get_client(fallback_role)
        except KeyError:
            raise ValueError(
                f"RuleBasedRoutingPolicy: fallback_role {fallback_role!r} is not "
                f"registered in the orchestrator. Register a ModelSpec with that "
                f"role or alias before creating RuleBasedRoutingPolicy."
            )
        self.orchestrator = orchestrator
        self.fallback_role = fallback_role
        self.n_per_dataset = n_per_dataset
        self._sem = asyncio.Semaphore(max_concurrent_questions)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample_cases(self, profiles: list[DatasetProfile]) -> list[EvaluationCase]:
        all_cases: list[EvaluationCase] = []
        for profile in profiles:
            bucket: list[EvaluationCase] = []
            try:
                for case in profile.iter_cases():
                    bucket.append(case)
                    if len(bucket) >= self.n_per_dataset:
                        break
            except Exception as exc:
                logger.warning("Skipping dataset %r — iteration failed: %s", profile.name, exc)
                continue
            all_cases.extend(bucket)
            logger.info("Sampled %d / %d from %s", len(bucket), self.n_per_dataset, profile.name)
        return all_cases

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, case: EvaluationCase) -> TaskType:
        dataset = case.example.dataset_name.lower()
        if dataset in _DATASET_TO_TASK:
            return _DATASET_TO_TASK[dataset]
        return classify_task(case.example.question or "")

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _build_answer_request(self, case: EvaluationCase) -> PromptRequest:
        example = case.example
        if example.messages:
            return PromptRequest(messages=example.messages)
        return PromptRequest(
            user_prompt=example.question,
            context=example.context,
            system_prompt=example.system_prompt,
        )

    async def _dispatch(
        self, task_type: TaskType, request: PromptRequest
    ) -> tuple[str, OrchestratorResponse]:
        """
        Dispatch to the specialist role for task_type, falling back to
        fallback_role if the specialist is not registered.

        Returns (routed_role, response).
        """
        role = TASK_TO_ROLE[task_type]
        try:
            response = await self.orchestrator.get_client(role).get_response(request)
            return role, response
        except KeyError:
            logger.warning(
                "Role %r not configured; falling back to %r", role, self.fallback_role
            )
            # fallback_role is validated at construction, so this KeyError
            # would only surface if the orchestrator was mutated after init.
            try:
                response = await self.orchestrator.get_client(
                    self.fallback_role
                ).get_response(request)
                return self.fallback_role, response
            except KeyError as exc:
                raise RuntimeError(
                    f"Primary role {role!r} and fallback role {self.fallback_role!r} "
                    f"are both missing from the orchestrator. "
                    f"Register at least one of them before running P3."
                ) from exc

    # ── Per-question orchestration ─────────────────────────────────────────────

    async def _run_question(self, case: EvaluationCase) -> P3QuestionResult | None:
        async with self._sem:
            task_type = self._classify(case)
            request = self._build_answer_request(case)
            try:
                routed_role, response = await self._dispatch(task_type, request)
                return P3QuestionResult(
                    case=case,
                    task_type=task_type,
                    routed_role=routed_role,
                    response=response,
                )
            except Exception as exc:
                logger.error(
                    "Unexpected error on %s: %s", case.example.example_id, exc, exc_info=True
                )
                return P3QuestionResult(
                    case=case,
                    task_type=task_type,
                    routed_role=self.fallback_role,
                    error=str(exc),
                )

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self, cases: list[EvaluationCase] | None = None
    ) -> P3PolicyResult:
        """
        Sample questions from all dataset profiles, route each to the
        best specialist, and return structured results.

        Args:
            profiles: Defaults to all available profiles when None.
        """
        # if profiles is None:
        #     profiles = load_all_profiles()

        # cases = self.sample_cases(profiles)
        # if not cases:
        #     logger.warning("No cases sampled — check that profiles are non-empty.")
        #     return P3PolicyResult()

        logger.info("P3 RuleBasedRoutingPolicy: %d questions, fallback=%r",len(cases),
                     self.fallback_role,
        )

        outcomes = await asyncio.gather(*[self._run_question(c) for c in cases])

        result = P3PolicyResult()
        for case, outcome in zip(cases, outcomes):
            if outcome is None or outcome.failed:
                result.skipped_question_ids.append(case.example.example_id)
                if outcome is not None:
                    result.results.append(outcome)
            else:
                result.results.append(outcome)

        result.routing_summaries = compute_routing_summary(result.results)

        logger.info(
            "P3 complete: %d succeeded, %d skipped.",
            sum(1 for r in result.results if not r.failed),
            len(result.skipped_question_ids),
        )
        for summary in result.routing_summaries:
            logger.info(
                "  %-20s  questions=%d  roles=%s  avg_latency=%.0fms",
                summary.dataset_name,
                summary.question_count,
                summary.role_counts,
                summary.avg_latency_ms or 0.0,
            )
        return result
