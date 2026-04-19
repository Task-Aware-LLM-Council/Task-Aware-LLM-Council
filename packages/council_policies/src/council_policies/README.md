# P2 Council Policy — Implementation Notes

This document describes every change made to `council_policies` to implement
the P2 dataset-level blind peer-review policy.

---

## Files changed

| File | Change |
|---|---|
| `p2_policy.py` | Full rewrite — `DatasetCouncilPolicy` |
| `prompts.py` | Added rater prompt section |
| `__init__.py` | Updated exports |

---

## 1. `prompts.py` — new rater section

Three additions at the bottom of the voter section:

### `RATER_SYSTEM_PROMPT`
System prompt given to each model when it acts as a rater. Instructs the
model to act as an impartial judge and score answers 1–10 based on accuracy,
completeness, and clarity. Explicitly says output must be JSON only — no
prose — so `parse_ratings` can reliably extract scores.

### `build_rating_prompt(question, labeled_answers) -> str`
Builds the user prompt shown to a rater. Reuses the same labeled-answer
layout as `build_voter_prompt` (Answer A / Answer B / Answer C blocks) but
appends a JSON schema instruction instead of "reply with a single letter".

```
Question:
<question text>

Answer A:
<answer text>

Answer B:
<answer text>

Answer C:
<answer text>

Rate every answer using exactly this JSON format:
{"A": {"score": <1-10>, "reasoning": "..."}, "B": {...}, "C": {...}}
```

### `parse_ratings(response_text, valid_labels) -> dict[str, float] | None`
Parses the model's JSON rating response into a `label → score` dict.

- **Attempt 1** — strict `json.loads`. Extracts `data[label]["score"]` for
  each label, clamps to 1–10.
- **Attempt 2 (fallback)** — regex scan per label:
  `"A": { ... "score": 7 ...` even if the surrounding JSON is malformed.
- Returns `None` if neither attempt yields all expected labels.
- Scores are always clamped to `[1.0, 10.0]` regardless of what the model
  returns.

---

## 2. `p2_policy.py` — full implementation

### Design

P2 is a **dataset-level blind peer-review council**. It is distinct from
the flat council (`FlatCouncilPolicy` / `voter.run_vote`) which operates on a
single prompt. P2:

1. Samples questions from all dataset profiles.
2. Sends the same question to all 3 council models in parallel.
3. Combines the 3 answers anonymously (shuffled labels A/B/C per rater).
4. Sends the combined answers back to all 3 models and asks for numerical
   ratings (1–10 JSON).
5. Aggregates ratings per dataset to produce a winner-per-dataset summary.

### Output models

#### `ModelAnswer`
One model's answer to one question.

| Field | Type | Notes |
|---|---|---|
| `role` | `str` | Orchestrator role name e.g. `"qa"` |
| `text` | `str` | Raw response text |
| `response` | `OrchestratorResponse \| None` | Full orchestrator response for provenance |
| `error` | `str \| None` | Set if generation failed |
| `failed` | `bool` (property) | True if `error` is set or `text` is empty |

#### `RatingEntry`
One score assigned by a rater to one labeled answer.

| Field | Type | Notes |
|---|---|---|
| `label` | `str` | `"A"`, `"B"`, or `"C"` as shown to this rater |
| `score` | `float` | 1–10 |
| `reasoning` | `str` | One-sentence justification from the model |

#### `RatingResult`
All ratings from one rater for one question.

| Field | Type | Notes |
|---|---|---|
| `rater_role` | `str` | Which model did the rating |
| `label_to_role` | `dict[str, str]` | Maps `"A"/"B"/"C"` → model role for this rater's shuffle |
| `ratings` | `list[RatingEntry]` | Empty if rating failed or parse failed |
| `raw_text` | `str` | Full model output (useful for debugging parse failures) |
| `error` | `str \| None` | Set if rating call or JSON parse failed |
| `failed` | `bool` (property) | True if `error` is set or `ratings` is empty |

#### `P2QuestionResult`
All council output for one question.

| Field | Type |
|---|---|
| `case` | `EvaluationCase` — original question + ground truth |
| `answers` | `list[ModelAnswer]` — one per council role |
| `ratings` | `list[RatingResult]` — one per rater (up to 3) |
| `best_answer` | `ModelAnswer \| None` (property) — winner for this specific question |

`best_answer` averages every score each model received across all raters for
this question and returns the `ModelAnswer` of the highest-scoring role.
Returns `None` only if all raters failed and no scores exist.

```python
qr = result.results[0]
print(qr.best_answer.role)   # e.g. "qa"
print(qr.best_answer.text)   # the actual winning answer text
```

This is different from `dataset_votes.winner` — `best_answer` is per question,
`dataset_votes.winner` is the model that performed best across all questions
in that dataset.

#### `DatasetVoteSummary`
Per-dataset aggregate across all questions.

| Field | Type | Notes |
|---|---|---|
| `dataset_name` | `str` | e.g. `"fever"`, `"hardmath"` |
| `scores_by_role` | `dict[str, float]` | Average score each model received |
| `winner` | `str` | Role with highest average score |
| `question_count` | `int` | How many questions contributed |

#### `P2PolicyResult`
Top-level return value of `DatasetCouncilPolicy.run()`.

| Field | Type |
|---|---|
| `results` | `list[P2QuestionResult]` |
| `skipped_question_ids` | `list[str]` — all 3 models failed on these |
| `dataset_votes` | `list[DatasetVoteSummary]` — sorted by dataset name |

### Anti-bias mechanism

A model asked to rate answers might prefer its own answer. To prevent this,
each rater gets a **different shuffle** of the A/B/C labels:

```
Rater "qa"        sees: A=qa's answer,        B=reasoning's, C=general's
Rater "reasoning" sees: A=reasoning's answer,  B=general's,   C=qa's
Rater "general"   sees: A=general's answer,    B=qa's,        C=reasoning's
```

The shuffle is **deterministic** (seeded by `seed` + `rater_index`) so runs
are reproducible. `label_to_role` in each `RatingResult` records the exact
mapping so the aggregator can decode scores back to model roles correctly.

### `compute_dataset_votes(results) -> list[DatasetVoteSummary]`

Standalone function (also exported from `__init__.py`) that aggregates
per-question ratings into per-dataset scores.

For every `P2QuestionResult`:
- For every non-failed `RatingResult`:
  - For every `RatingEntry`: decode label → role via `label_to_role`,
    append score to that role's list for that dataset.

Then per dataset: average each role's scores, pick the highest as `winner`.

### Edge cases handled

| Scenario | Behaviour |
|---|---|
| All 3 models fail to answer | Question skipped, ID added to `skipped_question_ids` |
| 1 or 2 models fail to answer | Proceeds — failed answers shown as `[NO RESPONSE]` to raters |
| Dataset iteration raises | Profile skipped with warning, others continue |
| Dataset has fewer than `n_per_dataset` questions | Takes all available |
| Role not registered in orchestrator | `ValueError` at `__init__` with clear message |
| Duplicate roles e.g. `("qa","qa","general")` | `ValueError` at `__init__` with clear message |
| Rating call raises exception | `RatingResult.error` set, `ratings=[]` |
| Rating response is not valid JSON | Regex fallback attempted; if both fail, `RatingResult.error` set |
| All raters fail for a question | Logged as warning; question still in `results` but contributes nothing to `dataset_votes` |
| Answer too long (code, essays) | Truncated to 2000 chars before combining in rating prompt |
| Question text is None or empty | Falls back to `"(no question text)"` in rating prompt |
| Unexpected exception per question | Caught, question skipped, run continues |
| No cases sampled at all | Returns empty `P2PolicyResult` immediately |

### Constants

| Name | Value | Purpose |
|---|---|---|
| `_DEFAULT_COUNCIL_ROLES` | `("qa", "reasoning", "general")` | Default 3 roles |
| `_MAX_ANSWER_CHARS` | `2000` | Max chars per answer in rating prompt |

### `load_all_profiles() -> list[DatasetProfile]`

Returns one instance of every available dataset profile:

| Profile | Dataset | Task type |
|---|---|---|
| `MusiqueProfile` | `bdsaglam/musique` | Multi-hop QA |
| `QualityProfile` | `narrativeqa` | Narrative QA |
| `FeverProfile` | `copenlu/fever_gold_evidence` | Fact verification |
| `HardMathProfile` | `math-ai/MATH-500` | Math |
| `HumanEvalPlusProfile` | `openai/openai_humaneval` | Code generation |

Used as the default when `profiles=None` is passed to `run()`.

### Usage

```python
from model_orchestration import ModelOrchestrator, build_default_orchestrator_config
from llm_gateway import Provider
from council_policies import DatasetCouncilPolicy

config = build_default_orchestrator_config(
    provider=Provider.OPENROUTER,
    api_key_env="OPENROUTER_API_KEY",
    qa_model="meta-llama/llama-3.1-8b-instruct",
    reasoning_model="deepseek/deepseek-r1",
    general_model="qwen/qwen-2.5-72b-instruct",
)

orchestrator = ModelOrchestrator(config)
async with orchestrator:
    policy = DatasetCouncilPolicy(
        orchestrator,
        council_roles=("qa", "reasoning", "general"),
        n_per_dataset=5,   # 5 questions x 5 datasets = 25 total
    )
    result = await policy.run()   # uses all 5 datasets by default

# Per-dataset winner
for summary in result.dataset_votes:
    print(f"{summary.dataset_name}: winner={summary.winner}")
    print(f"  scores: {summary.scores_by_role}")

# Best answer per question
for qr in result.results:
    best = qr.best_answer
    if best:
        print(f"{qr.case.example.example_id}: best model={best.role}, answer={best.text[:80]}")
```

---

## 3. `__init__.py` — updated exports

Added exports for all of `p2_policy.py` plus the previously missing
`LearnedRouterPolicy` from `p4_policy.py`.

### New exports

```python
from council_policies import (
    # P2
    DatasetCouncilPolicy,
    DatasetVoteSummary,
    ModelAnswer,
    P2PolicyResult,
    P2QuestionResult,
    RatingEntry,
    RatingResult,
    compute_dataset_votes,
    load_all_profiles,
    # P4 (was missing from __init__)
    LearnedRouterPolicy,
)
```

---

## What P2 does NOT do

- **Does not judge objective correctness.** Ratings are peer-perceived quality.
  Ground-truth scoring (`exact_match`, `token_f1`, `math_exact_match`) is the
  aggregator's responsibility using `task_eval`.
- **Does not synthesize answers.** That is `synthesis.synthesize` — used by
  P3/P4. P2 rates and selects, does not combine.
- **Does not talk to providers directly.** All calls go through
  `orchestrator.get_client(role).get_response(request)`.
