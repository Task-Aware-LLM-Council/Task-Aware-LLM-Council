"""
P2 Council Policy — Dataset-level blind peer-review.

Flow per question
-----------------
1. Fan out the question to all 3 council models concurrently.
2. Combine the 3 answers (labeled A/B/C, shuffled per rater) using
   build_rating_prompt from prompts.py.
3. Send the combined prompt back to all 3 models and ask for numerical
   ratings (1-10 JSON) — same models act as raters.
4. Aggregate ratings across all questions per dataset → dataset_votes
   tells you which model is best for each dataset type.

The aggregator (separate module) receives P2PolicyResult and decides
the final answer per question using the ratings.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field

from llm_gateway import PromptRequest
from model_orchestration import ModelOrchestrator, OrchestratorResponse
from task_eval.interfaces import DatasetProfile
from task_eval.models import EvaluationCase
from task_eval.profiles import (
    FeverProfile,
    HardMathProfile,
    HumanEvalPlusProfile,
    MusiqueProfile,
    QualityProfile,
)

from council_policies.prompts import (
    RATER_SYSTEM_PROMPT,
    build_rating_prompt,
    parse_ratings,
)

logger = logging.getLogger(__name__)

_LABELS: tuple[str, ...] = ("A", "B", "C")

_DEFAULT_COUNCIL_ROLES = ("qa", "reasoning", "general")

# Max characters per answer shown to raters — prevents token limit issues
# when combining 3 long answers (code, essays) in one rating prompt.
_MAX_ANSWER_CHARS = 2000


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_all_profiles() -> list[DatasetProfile]:
    """Return one instance of every available dataset profile."""
    return [
        MusiqueProfile(),
        QualityProfile(),
        FeverProfile(),
        HardMathProfile(),
        HumanEvalPlusProfile(),
    ]


# ── Output models ──────────────────────────────────────────────────────────────

@dataclass
class ModelAnswer:
    role: str
    text: str
    response: OrchestratorResponse | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return bool(self.error) or not self.text.strip()


@dataclass
class RatingEntry:
    label: str          # "A", "B", or "C" as shown to this rater
    score: float        # 1–10
    reasoning: str = ""


@dataclass
class RatingResult:
    rater_role: str
    label_to_role: dict[str, str]   # lets aggregator decode label → model role
    ratings: list[RatingEntry]
    raw_text: str = ""
    error: str | None = None

    @property
    def failed(self) -> bool:
        return bool(self.error) or not self.ratings


@dataclass
class P2QuestionResult:
    case: EvaluationCase
    answers: list[ModelAnswer]    # one per council role
    ratings: list[RatingResult]   # one per rater role

    @property
    def best_answer(self) -> ModelAnswer | None:
        """
        The answer that received the highest average score across all raters
        for this specific question. Returns None if no usable ratings exist.
        """
        role_scores: dict[str, list[float]] = {}
        for rating_result in self.ratings:
            if rating_result.failed:
                continue
            for entry in rating_result.ratings:
                role = rating_result.label_to_role.get(entry.label)
                if role is None:
                    continue
                role_scores.setdefault(role, []).append(entry.score)

        if not role_scores:
            return None

        avg_scores = {role: sum(s) / len(s) for role, s in role_scores.items()}
        best_role = max(avg_scores, key=lambda r: avg_scores[r])

        for answer in self.answers:
            if answer.role == best_role:
                return answer
        return None


@dataclass
class DatasetVoteSummary:
    """Per-dataset aggregate: which model scored highest across all questions."""

    dataset_name: str
    scores_by_role: dict[str, float]   # role → average score
    winner: str
    question_count: int


@dataclass
class P2PolicyResult:
    results: list[P2QuestionResult] = field(default_factory=list)
    skipped_question_ids: list[str] = field(default_factory=list)
    dataset_votes: list[DatasetVoteSummary] = field(default_factory=list)


# ── Dataset-level vote aggregation ────────────────────────────────────────────

def compute_dataset_votes(results: list[P2QuestionResult]) -> list[DatasetVoteSummary]:
    """
    Aggregate per-question ratings into per-dataset model scores.
    For each dataset, averages every score each model received across
    all questions and all raters, then picks the winner.
    """
    totals: dict[str, dict[str, list[float]]] = {}

    for qr in results:
        dataset = qr.case.example.dataset_name
        if dataset not in totals:
            totals[dataset] = {}
        for rating_result in qr.ratings:
            if rating_result.failed:
                continue
            for entry in rating_result.ratings:
                role = rating_result.label_to_role.get(entry.label)
                if role is None:
                    continue
                totals[dataset].setdefault(role, []).append(entry.score)

    summaries: list[DatasetVoteSummary] = []
    for dataset, role_scores in totals.items():
        if not role_scores:
            continue
        avg = {role: sum(s) / len(s) for role, s in role_scores.items()}
        winner = max(avg, key=lambda r: avg[r])
        question_count = sum(
            1 for qr in results if qr.case.example.dataset_name == dataset
        )
        summaries.append(DatasetVoteSummary(
            dataset_name=dataset,
            scores_by_role=avg,
            winner=winner,
            question_count=question_count,
        ))

    summaries.sort(key=lambda s: s.dataset_name)
    return summaries


# ── P2 Policy ──────────────────────────────────────────────────────────────────

class DatasetCouncilPolicy:
    """
    P2: Dataset council with blind peer-review rating.

    Usage::

        orchestrator = ModelOrchestrator(config)
        async with orchestrator:
            policy = DatasetCouncilPolicy(orchestrator)
            result = await policy.run()   # samples all datasets automatically
    """

    def __init__(
        self,
        orchestrator: ModelOrchestrator,
        *,
        council_roles: tuple[str, ...] = _DEFAULT_COUNCIL_ROLES,
        n_per_dataset: int = 5,
        max_concurrent_questions: int = 4,
        seed: int = 42,
    ) -> None:
        if len(council_roles) != 3:
            raise ValueError(
                f"DatasetCouncilPolicy requires exactly 3 council_roles, got {len(council_roles)}"
            )
        if len(set(council_roles)) != 3:
            raise ValueError(f"council_roles must be distinct, got duplicates: {council_roles}")
        # Validate all roles are registered — fail fast rather than at inference time
        for role in council_roles:
            try:
                orchestrator.get_client(role)
            except KeyError:
                raise ValueError(
                    f"Role {role!r} is not registered in the orchestrator. "
                    f"Register a ModelSpec with that role before creating DatasetCouncilPolicy."
                )
        self.orchestrator = orchestrator
        self.council_roles = council_roles
        self.n_per_dataset = n_per_dataset
        self._sem = asyncio.Semaphore(max_concurrent_questions)
        self._rng = random.Random(seed)

    # ── Sampling ───────────────────────────────────────────────────────────────

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

    # ── Generation phase ───────────────────────────────────────────────────────

    def _build_answer_request(self, case: EvaluationCase) -> PromptRequest:
        example = case.example
        if example.messages:
            return PromptRequest(messages=example.messages)
        return PromptRequest(
            user_prompt=example.question,
            context=example.context,
            system_prompt=example.system_prompt,
        )

    async def _get_answer(self, role: str, request: PromptRequest) -> ModelAnswer:
        try:
            response = await self.orchestrator.get_client(role).get_response(request)
            text = response.text.strip()
            if not text:
                return ModelAnswer(role=role, text="", error="empty response")
            return ModelAnswer(role=role, text=text, response=response)
        except Exception as exc:
            logger.warning("Role %r failed to answer: %s", role, exc)
            return ModelAnswer(role=role, text="", error=str(exc))

    async def _gather_answers(self, case: EvaluationCase) -> list[ModelAnswer]:
        request = self._build_answer_request(case)
        return list(await asyncio.gather(
            *[self._get_answer(role, request) for role in self.council_roles]
        ))

    # ── Rating phase ───────────────────────────────────────────────────────────

    def _shuffle_labels(
        self, answers: list[ModelAnswer], rater_index: int
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Assign shuffled A/B/C labels to the answers for this rater.
        Each rater gets a distinct shuffle (seeded) to reduce position bias.

        Returns:
            labeled_answers  – label → answer text (for build_rating_prompt)
            label_to_role    – label → role (for aggregator to decode scores)
        """
        local_rng = random.Random(self._rng.random() + rater_index * 1_000_003)
        order = list(range(len(answers)))
        local_rng.shuffle(order)

        labeled_answers: dict[str, str] = {}
        label_to_role: dict[str, str] = {}
        for label, idx in zip(_LABELS, order):
            ans = answers[idx]
            if ans.failed:
                text = "[NO RESPONSE]"
            else:
                # Truncate to avoid exceeding token limits when 3 long answers are combined
                text = ans.text if len(ans.text) <= _MAX_ANSWER_CHARS else ans.text[:_MAX_ANSWER_CHARS] + "…[truncated]"
            labeled_answers[label] = text
            label_to_role[label] = ans.role
        return labeled_answers, label_to_role

    async def _get_rating(
        self,
        rater_index: int,
        case: EvaluationCase,
        answers: list[ModelAnswer],
    ) -> RatingResult:
        rater_role = self.council_roles[rater_index]
        labeled_answers, label_to_role = self._shuffle_labels(answers, rater_index)

        # build_rating_prompt from prompts.py combines the 3 answers and asks for JSON scores
        # Use first 500 chars of question in rating prompt to keep it readable
        question = (case.example.question or "")[:500] or "(no question text)"
        request = PromptRequest(
            system_prompt=RATER_SYSTEM_PROMPT,
            user_prompt=build_rating_prompt(question, labeled_answers),
        )

        try:
            response = await self.orchestrator.get_client(rater_role).get_response(request)
            raw = response.text.strip()
            scores = parse_ratings(raw, list(_LABELS))
            if scores is None:
                return RatingResult(
                    rater_role=rater_role,
                    label_to_role=label_to_role,
                    ratings=[],
                    raw_text=raw,
                    error="could not parse rating JSON",
                )
            return RatingResult(
                rater_role=rater_role,
                label_to_role=label_to_role,
                ratings=[RatingEntry(label=lbl, score=score) for lbl, score in scores.items()],
                raw_text=raw,
            )
        except Exception as exc:
            logger.warning("Rater %r failed on %s: %s", rater_role, case.example.example_id, exc)
            return RatingResult(
                rater_role=rater_role,
                label_to_role=label_to_role,
                ratings=[],
                error=str(exc),
            )

    async def _gather_ratings(
        self, case: EvaluationCase, answers: list[ModelAnswer]
    ) -> list[RatingResult]:
        return list(await asyncio.gather(
            *[self._get_rating(i, case, answers) for i in range(len(self.council_roles))]
        ))

    # ── Per-question orchestration ─────────────────────────────────────────────

    async def _run_question(self, case: EvaluationCase) -> P2QuestionResult | None:
        async with self._sem:
            try:
                answers = await self._gather_answers(case)

                if all(a.failed for a in answers):
                    logger.error("All models failed for %s — skipping.", case.example.example_id)
                    return None

                n_failed = sum(a.failed for a in answers)
                if n_failed:
                    logger.warning(
                        "%d / %d models failed for %s; proceeding with partial answers.",
                        n_failed, len(self.council_roles), case.example.example_id,
                    )

                ratings = await self._gather_ratings(case, answers)

                if all(r.failed for r in ratings):
                    logger.warning(
                        "All raters failed for %s — question has no usable ratings.",
                        case.example.example_id,
                    )

                return P2QuestionResult(case=case, answers=answers, ratings=ratings)

            except Exception as exc:
                logger.error(
                    "Unexpected error on %s: %s", case.example.example_id, exc, exc_info=True
                )
                return None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        profiles: list[DatasetProfile] | None = None,
        cases: list[EvaluationCase] | None = None,
    ) -> P2PolicyResult:
        """
        Run generation + peer-review rating and return structured results.

        Args:
            cases:    Pass a list of EvaluationCase directly — skips dataset
                      loading entirely. Use this to run P2 on the same questions
                      as P1 or any other source.
            profiles: Used only when `cases` is None. Defaults to all available
                      profiles when both are None.
        """
        if cases is not None:
            pass  # use provided cases directly
        else:
            if profiles is None:
                profiles = load_all_profiles()
            cases = self.sample_cases(profiles)
        if not cases:
            logger.warning("No cases sampled — check that profiles are non-empty.")
            return P2PolicyResult()

        logger.info(
            "P2 DatasetCouncilPolicy: %d questions, %d datasets, roles=%s",
            len(cases), len(profiles) if profiles is not None else "?", self.council_roles,
        )

        outcomes = await asyncio.gather(*[self._run_question(c) for c in cases])

        result = P2PolicyResult()
        for case, outcome in zip(cases, outcomes):
            if outcome is None:
                result.skipped_question_ids.append(case.example.example_id)
            else:
                result.results.append(outcome)

        result.dataset_votes = compute_dataset_votes(result.results)

        logger.info(
            "P2 complete: %d succeeded, %d skipped.", len(result.results), len(result.skipped_question_ids)
        )
        for summary in result.dataset_votes:
            logger.info(
                "  %-20s → winner: %-12s | %s",
                summary.dataset_name,
                summary.winner,
                {r: f"{s:.1f}" for r, s in summary.scores_by_role.items()},
            )
        return result
