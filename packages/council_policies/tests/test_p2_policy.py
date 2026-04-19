"""
Tests for P2 council policy — DatasetCouncilPolicy.

Uses a FakeOrchestrator so no real API calls are made.
Run with: uv run pytest packages/council_policies/tests/test_p2_policy.py -v
"""

from __future__ import annotations

import pytest

from llm_gateway import PromptRequest, PromptResponse, ProviderConfig, Provider, Usage
from model_orchestration import ModelOrchestrator, ModelSpec, OrchestratorConfig

from council_policies.p2_policy import (
    DatasetCouncilPolicy,
    DatasetVoteSummary,
    ModelAnswer,
    P2PolicyResult,
    P2QuestionResult,
    RatingEntry,
    RatingResult,
    compute_dataset_votes,
    load_all_profiles,
)
from council_policies.prompts import (
    build_rating_prompt,
    parse_ratings,
    RATER_SYSTEM_PROMPT,
)
from task_eval.models import EvaluationCase
from benchmarking_pipeline import BenchmarkExample


# ── Fake infrastructure ────────────────────────────────────────────────────────

_DEFAULT_RATING_JSON = '{"A": {"score": 8, "reasoning": "good"}, "B": {"score": 6, "reasoning": "ok"}, "C": {"score": 7, "reasoning": "decent"}}'


class FakeClient:
    """
    Mimics BaseLLMClient.

    Returns `answer_text` for answer-phase calls and `rating_text` for rating-phase
    calls (detected by RATER_SYSTEM_PROMPT in the request).
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        answer_text: str = "fake answer",
        rating_text: str = _DEFAULT_RATING_JSON,
        fail: bool = False,
    ) -> None:
        self.config = config
        self._answer_text = answer_text
        self._rating_text = rating_text
        self._fail = fail
        self.calls: list[PromptRequest] = []

    async def generate(self, request: PromptRequest) -> PromptResponse:
        self.calls.append(request)
        if self._fail:
            raise RuntimeError("deliberate fake failure")
        is_rating = request.system_prompt == RATER_SYSTEM_PROMPT
        text = self._rating_text if is_rating else self._answer_text
        return PromptResponse(
            model=self.config.default_model or "fake-model",
            text=text,
            usage=Usage(input_tokens=5, output_tokens=5, total_tokens=10),
        )

    async def close(self) -> None:
        pass


def _make_orchestrator(
    qa_text: str = "qa answer",
    reasoning_text: str = "reasoning answer",
    general_text: str = "general answer",
    qa_fail: bool = False,
    reasoning_fail: bool = False,
    general_fail: bool = False,
    rating_text: str = _DEFAULT_RATING_JSON,
) -> ModelOrchestrator:
    """Build an orchestrator backed entirely by FakeClients."""

    texts = {"qa": qa_text, "reasoning": reasoning_text, "general": general_text}
    fails = {"qa": qa_fail, "reasoning": reasoning_fail, "general": general_fail}

    def client_builder(config: ProviderConfig) -> FakeClient:
        model = config.default_model or ""
        for role in ("qa", "reasoning", "general"):
            if role in model:
                return FakeClient(config, answer_text=texts[role], rating_text=rating_text, fail=fails[role])
        return FakeClient(config, answer_text="unknown", rating_text=rating_text)

    config = OrchestratorConfig(
        models=(
            ModelSpec(
                role="qa",
                model="qa-model",
                aliases=("qa",),
                provider_config=ProviderConfig(
                    provider=Provider.OPENAI_COMPATIBLE,
                    api_base="http://fake/v1/chat/completions",
                    default_model="qa-model",
                ),
            ),
            ModelSpec(
                role="reasoning",
                model="reasoning-model",
                aliases=("reasoning",),
                provider_config=ProviderConfig(
                    provider=Provider.OPENAI_COMPATIBLE,
                    api_base="http://fake/v1/chat/completions",
                    default_model="reasoning-model",
                ),
            ),
            ModelSpec(
                role="general",
                model="general-model",
                aliases=("general",),
                provider_config=ProviderConfig(
                    provider=Provider.OPENAI_COMPATIBLE,
                    api_base="http://fake/v1/chat/completions",
                    default_model="general-model",
                ),
            ),
        ),
        default_role="general",
    )
    return ModelOrchestrator(config, client_builder=client_builder)


def _make_case(example_id: str = "ex-1", dataset: str = "musique", question: str = "What is 2+2?") -> EvaluationCase:
    example = BenchmarkExample(
        example_id=example_id,
        dataset_name=dataset,
        question=question,
    )
    return EvaluationCase(example=example)


# ── prompts.py unit tests ──────────────────────────────────────────────────────

class TestParseRatings:
    def test_valid_json(self):
        raw = '{"A": {"score": 8, "reasoning": "good"}, "B": {"score": 5, "reasoning": "ok"}, "C": {"score": 9, "reasoning": "great"}}'
        result = parse_ratings(raw, ["A", "B", "C"])
        assert result == {"A": 8.0, "B": 5.0, "C": 9.0}

    def test_clamps_above_10(self):
        raw = '{"A": {"score": 15, "reasoning": ""}, "B": {"score": 0, "reasoning": ""}, "C": {"score": 5, "reasoning": ""}}'
        result = parse_ratings(raw, ["A", "B", "C"])
        assert result["A"] == 10.0
        assert result["B"] == 1.0

    def test_fallback_regex(self):
        # Malformed JSON but regex can extract scores
        raw = '{"A": {"score": 7}, "B": {"score": 4}, "C": {"score": 9}}'
        result = parse_ratings(raw, ["A", "B", "C"])
        assert result is not None
        assert result["C"] == 9.0

    def test_returns_none_for_garbage(self):
        assert parse_ratings("I cannot rate this.", ["A", "B", "C"]) is None

    def test_strips_markdown_code_fences(self):
        raw = "```json\n{\"A\": {\"score\": 7, \"reasoning\": \"\"}, \"B\": {\"score\": 6, \"reasoning\": \"\"}, \"C\": {\"score\": 5, \"reasoning\": \"\"}}\n```"
        result = parse_ratings(raw, ["A", "B", "C"])
        assert result == {"A": 7.0, "B": 6.0, "C": 5.0}

    def test_missing_label_returns_none(self):
        raw = '{"A": {"score": 8, "reasoning": "good"}, "B": {"score": 5, "reasoning": "ok"}}'
        assert parse_ratings(raw, ["A", "B", "C"]) is None


class TestBuildRatingPrompt:
    def test_contains_question_and_answers(self):
        prompt = build_rating_prompt("What is the capital?", {"A": "Paris", "B": "London", "C": "Berlin"})
        assert "What is the capital?" in prompt
        assert "Answer A:" in prompt
        assert "Paris" in prompt
        assert "Answer B:" in prompt
        assert "Answer C:" in prompt

    def test_contains_json_schema_instruction(self):
        prompt = build_rating_prompt("Q?", {"A": "a", "B": "b", "C": "c"})
        assert "score" in prompt
        assert "reasoning" in prompt


# ── ModelAnswer / RatingResult unit tests ─────────────────────────────────────

class TestModelAnswer:
    def test_failed_when_error_set(self):
        a = ModelAnswer(role="qa", text="", error="timeout")
        assert a.failed is True

    def test_failed_when_empty_text(self):
        a = ModelAnswer(role="qa", text="   ")
        assert a.failed is True

    def test_not_failed_when_text_present(self):
        a = ModelAnswer(role="qa", text="some answer")
        assert a.failed is False


class TestRatingResult:
    def test_failed_when_error_set(self):
        r = RatingResult(rater_role="qa", label_to_role={}, ratings=[], error="boom")
        assert r.failed is True

    def test_failed_when_no_ratings(self):
        r = RatingResult(rater_role="qa", label_to_role={}, ratings=[])
        assert r.failed is True

    def test_not_failed_when_ratings_present(self):
        r = RatingResult(
            rater_role="qa",
            label_to_role={"A": "qa"},
            ratings=[RatingEntry(label="A", score=8.0)],
        )
        assert r.failed is False


# ── P2QuestionResult.best_answer ───────────────────────────────────────────────

class TestBestAnswer:
    def _make_result(self, scores: dict[str, float]) -> P2QuestionResult:
        answers = [
            ModelAnswer(role="qa", text="qa answer"),
            ModelAnswer(role="reasoning", text="reasoning answer"),
            ModelAnswer(role="general", text="general answer"),
        ]
        label_to_role = {"A": "qa", "B": "reasoning", "C": "general"}
        ratings = [
            RatingResult(
                rater_role="qa",
                label_to_role=label_to_role,
                ratings=[
                    RatingEntry(label=lbl, score=score)
                    for lbl, score in [("A", scores["qa"]), ("B", scores["reasoning"]), ("C", scores["general"])]
                ],
            )
        ]
        return P2QuestionResult(case=_make_case(), answers=answers, ratings=ratings)

    def test_returns_highest_scoring_role(self):
        qr = self._make_result({"qa": 5.0, "reasoning": 9.0, "general": 7.0})
        assert qr.best_answer is not None
        assert qr.best_answer.role == "reasoning"

    def test_returns_none_when_all_raters_failed(self):
        answers = [ModelAnswer(role="qa", text="answer")]
        ratings = [RatingResult(rater_role="qa", label_to_role={}, ratings=[], error="failed")]
        qr = P2QuestionResult(case=_make_case(), answers=answers, ratings=ratings)
        assert qr.best_answer is None

    def test_averages_across_multiple_raters(self):
        answers = [
            ModelAnswer(role="qa", text="qa answer"),
            ModelAnswer(role="reasoning", text="reasoning answer"),
            ModelAnswer(role="general", text="general answer"),
        ]
        ratings = [
            RatingResult(
                rater_role="qa",
                label_to_role={"A": "qa", "B": "reasoning", "C": "general"},
                ratings=[RatingEntry("A", 9.0), RatingEntry("B", 5.0), RatingEntry("C", 5.0)],
            ),
            RatingResult(
                rater_role="reasoning",
                label_to_role={"A": "reasoning", "B": "general", "C": "qa"},
                ratings=[RatingEntry("A", 3.0), RatingEntry("B", 5.0), RatingEntry("C", 3.0)],
            ),
        ]
        qr = P2QuestionResult(case=_make_case(), answers=answers, ratings=ratings)
        # qa avg = (9+3)/2=6, reasoning avg = (5+3)/2=4, general avg = (5+5)/2=5
        assert qr.best_answer.role == "qa"


# ── compute_dataset_votes ──────────────────────────────────────────────────────

class TestComputeDatasetVotes:
    def _make_qr(self, dataset: str, scores: dict[str, float], example_id: str = "ex-1") -> P2QuestionResult:
        answers = [ModelAnswer(role=r, text="t") for r in scores]
        label_to_role = {chr(65 + i): role for i, role in enumerate(scores)}
        ratings = [
            RatingResult(
                rater_role="qa",
                label_to_role=label_to_role,
                ratings=[
                    RatingEntry(label=chr(65 + i), score=s)
                    for i, s in enumerate(scores.values())
                ],
            )
        ]
        return P2QuestionResult(case=_make_case(example_id, dataset), answers=answers, ratings=ratings)

    def test_picks_correct_winner(self):
        results = [self._make_qr("musique", {"qa": 9.0, "reasoning": 5.0, "general": 7.0})]
        summaries = compute_dataset_votes(results)
        assert len(summaries) == 1
        assert summaries[0].dataset_name == "musique"
        assert summaries[0].winner == "qa"

    def test_aggregates_multiple_questions(self):
        results = [
            self._make_qr("musique", {"qa": 3.0, "reasoning": 9.0}, "q1"),
            self._make_qr("musique", {"qa": 9.0, "reasoning": 3.0}, "q2"),
        ]
        summaries = compute_dataset_votes(results)
        # qa avg=6, reasoning avg=6 — tie broken by max() which is stable (qa comes first)
        assert summaries[0].question_count == 2

    def test_separates_datasets(self):
        results = [
            self._make_qr("musique", {"qa": 8.0}, "q1"),
            self._make_qr("fever", {"reasoning": 9.0}, "q2"),
        ]
        summaries = compute_dataset_votes(results)
        names = {s.dataset_name for s in summaries}
        assert names == {"musique", "fever"}

    def test_sorted_by_dataset_name(self):
        results = [
            self._make_qr("zebra", {"qa": 5.0}, "q1"),
            self._make_qr("apple", {"qa": 5.0}, "q2"),
        ]
        summaries = compute_dataset_votes(results)
        assert summaries[0].dataset_name == "apple"
        assert summaries[1].dataset_name == "zebra"

    def test_skips_failed_raters(self):
        answers = [ModelAnswer(role="qa", text="t")]
        ratings = [RatingResult(rater_role="qa", label_to_role={}, ratings=[], error="failed")]
        qr = P2QuestionResult(case=_make_case(dataset="musique"), answers=answers, ratings=ratings)
        summaries = compute_dataset_votes([qr])
        assert summaries == []


# ── DatasetCouncilPolicy init validation ──────────────────────────────────────

class TestDatasetCouncilPolicyInit:
    def test_rejects_wrong_number_of_roles(self):
        orchestrator = _make_orchestrator()
        with pytest.raises(ValueError, match="exactly 3"):
            DatasetCouncilPolicy(orchestrator, council_roles=("qa", "reasoning"))

    def test_rejects_duplicate_roles(self):
        orchestrator = _make_orchestrator()
        with pytest.raises(ValueError, match="distinct"):
            DatasetCouncilPolicy(orchestrator, council_roles=("qa", "qa", "general"))

    def test_rejects_unknown_role(self):
        orchestrator = _make_orchestrator()
        with pytest.raises(ValueError, match="not registered"):
            DatasetCouncilPolicy(orchestrator, council_roles=("qa", "reasoning", "unknown_role"))

    def test_accepts_valid_roles(self):
        orchestrator = _make_orchestrator()
        policy = DatasetCouncilPolicy(orchestrator)
        assert policy.council_roles == ("qa", "reasoning", "general")


# ── DatasetCouncilPolicy._shuffle_labels ──────────────────────────────────────

class TestShuffleLabels:
    def test_all_roles_assigned(self):
        orchestrator = _make_orchestrator()
        policy = DatasetCouncilPolicy(orchestrator, seed=0)
        answers = [
            ModelAnswer(role="qa", text="a"),
            ModelAnswer(role="reasoning", text="b"),
            ModelAnswer(role="general", text="c"),
        ]
        labeled, label_to_role = policy._shuffle_labels(answers, 0)
        assert set(labeled.keys()) == {"A", "B", "C"}
        assert set(label_to_role.values()) == {"qa", "reasoning", "general"}

    def test_different_raters_get_different_shuffles(self):
        orchestrator = _make_orchestrator()
        policy = DatasetCouncilPolicy(orchestrator, seed=0)
        answers = [
            ModelAnswer(role="qa", text="a"),
            ModelAnswer(role="reasoning", text="b"),
            ModelAnswer(role="general", text="c"),
        ]
        maps = [policy._shuffle_labels(answers, i)[1] for i in range(3)]
        # Not all three shuffles should be identical
        assert not (maps[0] == maps[1] == maps[2])

    def test_failed_answer_shown_as_no_response(self):
        orchestrator = _make_orchestrator()
        policy = DatasetCouncilPolicy(orchestrator)
        answers = [
            ModelAnswer(role="qa", text="", error="timeout"),
            ModelAnswer(role="reasoning", text="good answer"),
            ModelAnswer(role="general", text="another answer"),
        ]
        labeled, label_to_role = policy._shuffle_labels(answers, 0)
        failed_label = next(lbl for lbl, role in label_to_role.items() if role == "qa")
        assert labeled[failed_label] == "[NO RESPONSE]"

    def test_long_answers_truncated(self):
        orchestrator = _make_orchestrator()
        policy = DatasetCouncilPolicy(orchestrator)
        long_text = "x" * 3000
        answers = [
            ModelAnswer(role="qa", text=long_text),
            ModelAnswer(role="reasoning", text="short"),
            ModelAnswer(role="general", text="short"),
        ]
        labeled, label_to_role = policy._shuffle_labels(answers, 0)
        qa_label = next(lbl for lbl, role in label_to_role.items() if role == "qa")
        assert len(labeled[qa_label]) < 3000
        assert "truncated" in labeled[qa_label]


# ── Full async run with fake orchestrator ─────────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_result_with_answers_and_ratings():
    orchestrator = _make_orchestrator(
        qa_text="qa says Paris",
        reasoning_text="reasoning says Paris",
        general_text="general says Paris",
        rating_text='{"A": {"score": 8, "reasoning": "good"}, "B": {"score": 7, "reasoning": "ok"}, "C": {"score": 6, "reasoning": "meh"}}',
    )
    policy = DatasetCouncilPolicy(orchestrator, n_per_dataset=1)

    case = _make_case("q1", "musique", "What city is the capital of France?")

    # Run just one question directly
    result = await policy._run_question(case)

    assert result is not None
    assert len(result.answers) == 3
    assert all(not a.failed for a in result.answers)
    assert len(result.ratings) == 3
    assert result.best_answer is not None


@pytest.mark.asyncio
async def test_run_skips_question_when_all_models_fail():
    orchestrator = _make_orchestrator(qa_fail=True, reasoning_fail=True, general_fail=True)
    policy = DatasetCouncilPolicy(orchestrator, n_per_dataset=1)
    case = _make_case("q1", "musique")

    result = await policy._run_question(case)
    assert result is None


@pytest.mark.asyncio
async def test_run_continues_with_partial_failures():
    orchestrator = _make_orchestrator(
        qa_fail=True,
        reasoning_text="reasoning answer",
        general_text="general answer",
        rating_text='{"A": {"score": 5, "reasoning": ""}, "B": {"score": 8, "reasoning": ""}, "C": {"score": 6, "reasoning": ""}}',
    )
    policy = DatasetCouncilPolicy(orchestrator, n_per_dataset=1)
    case = _make_case("q1", "musique")

    result = await policy._run_question(case)
    assert result is not None
    failed = [a for a in result.answers if a.failed]
    assert len(failed) == 1
    assert failed[0].role == "qa"


# ── load_all_profiles smoke test ──────────────────────────────────────────────

def test_load_all_profiles_returns_five_profiles():
    profiles = load_all_profiles()
    assert len(profiles) == 5
    names = {p.name for p in profiles}
    assert "musique" in names


# ── run() cases= parameter ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_accepts_cases_directly():
    """run(cases=...) skips dataset loading and uses provided cases."""
    orchestrator = _make_orchestrator(
        rating_text='{"A": {"score": 8, "reasoning": "good"}, "B": {"score": 7, "reasoning": "ok"}, "C": {"score": 6, "reasoning": "meh"}}',
    )
    policy = DatasetCouncilPolicy(orchestrator, n_per_dataset=99)

    cases = [
        _make_case("q1", "custom", "What is 2+2?"),
        _make_case("q2", "custom", "Who wrote Hamlet?"),
    ]

    result = await policy.run(cases=cases)

    assert len(result.results) + len(result.skipped_question_ids) == 2
    # dataset_votes should be for "custom" dataset
    assert any(s.dataset_name == "custom" for s in result.dataset_votes)


@pytest.mark.asyncio
async def test_run_cases_overrides_profiles():
    """When cases= is provided, profiles= is ignored entirely."""
    orchestrator = _make_orchestrator(
        rating_text='{"A": {"score": 9, "reasoning": ""}, "B": {"score": 7, "reasoning": ""}, "C": {"score": 5, "reasoning": ""}}',
    )
    policy = DatasetCouncilPolicy(orchestrator, n_per_dataset=99)

    cases = [_make_case("q1", "override_dataset", "Test question")]

    # Pass a profiles list too — should be ignored because cases= is set
    result = await policy.run(cases=cases, profiles=[])

    assert len(result.results) + len(result.skipped_question_ids) == 1


# ── p2_benchmark smoke test ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_p2_benchmark_saves_output(tmp_path):
    """run_p2_benchmark_suite writes score files and suite_metrics.json."""
    from council_policies.p2_benchmark import run_p2_benchmark_suite

    orchestrator = _make_orchestrator(
        rating_text='{"A": {"score": 8, "reasoning": "good"}, "B": {"score": 6, "reasoning": "ok"}, "C": {"score": 7, "reasoning": "decent"}}',
    )

    # Fake DatasetSource with 2 questions
    class FakeSource:
        name = "fake_dataset"
        metadata = {}

        def iter_cases(self):
            yield _make_case("q1", "fake_dataset", "Question one?")
            yield _make_case("q2", "fake_dataset", "Question two?")

    result = await run_p2_benchmark_suite(
        [FakeSource()],
        orchestrator,
        output_root=tmp_path,
        n_per_dataset=5,
    )

    assert result.total_examples == 2
    assert result.aggregate_summary_path.exists()
    assert len(result.score_files) == 1
    assert result.score_files[0].exists()

    # score file should have 2 lines
    lines = result.score_files[0].read_text().strip().splitlines()
    assert len(lines) == 2
