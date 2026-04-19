# P2 Council Policy — Implementation Notes

This document describes every change made to `council_policies` to implement
the P2 dataset-level blind peer-review policy and the supporting benchmark
infrastructure to compare P1 (individual models) vs P2 (council).

---

## Files changed / added

| File | Change |
|---|---|
| `p2_policy.py` | Full implementation — `DatasetCouncilPolicy`; `cases=` param added to `run()` |
| `prompts.py` | Added rater prompt section (`RATER_SYSTEM_PROMPT`, `build_rating_prompt`, `parse_ratings`) |
| `run.py` | CLI entry point (`council-p2`); nvidia provider support; `--output` JSON saving; `--api-base` override |
| `p2_benchmark.py` | New — benchmark runner scoring council output with task-eval metrics |
| `bench_cli.py` | New — CLI entry point (`council-p2-bench`); nvidia provider support; `metric_resolver` wired |
| `p1_cli.py` | New — CLI entry point (`council-p1`) for running individual model benchmarks |
| `policy_adapters.py` | Added `P2DatasetAdapter` — plugs P2 blind peer-review into `CouncilBenchmarkRunner` framework |
| `task_eval/profiles.py` | Fixed `HardMathProfile` and `HumanEvalPlusProfile` to use `split="test"` |
| `pyproject.toml` | Registered `council-p1`, `council-p2`, `council-p2-bench` entry points |

---

## 1. `prompts.py` — rater section

### `RATER_SYSTEM_PROMPT`
System prompt given to each model when acting as a rater. Instructs the model
to act as an impartial judge and score answers 1–10 based on accuracy,
completeness, and clarity. Output must be JSON only.

### `build_rating_prompt(question, labeled_answers) -> str`
Builds the user prompt for a rater. Shows 3 answers labeled A/B/C and asks
for scores in a strict JSON format:
```
{"A": {"score": <1-10>, "reasoning": "..."}, "B": {...}, "C": {...}}
```

### `parse_ratings(response_text, valid_labels) -> dict[str, float] | None`
Parses the model's JSON rating response into `label → score` dict.
- Attempt 1: strict `json.loads`
- Attempt 2 (fallback): regex scan per label for malformed JSON
- Scores clamped to `[1.0, 10.0]`
- Returns `None` if neither attempt yields all expected labels

---

## 2. `p2_policy.py` — full implementation

### Design

P2 is a **dataset-level blind peer-review council**:

1. Samples questions from all 5 dataset profiles.
2. Sends the same question to all 3 council models in parallel.
3. Combines the 3 answers anonymously (shuffled A/B/C labels per rater).
4. Sends combined answers back to all 3 models for numerical ratings (1–10 JSON).
5. Aggregates ratings per dataset → `dataset_votes` tells you which model is best per dataset.

### Output models

#### `ModelAnswer`
| Field | Type | Notes |
|---|---|---|
| `role` | `str` | Orchestrator role e.g. `"qa"` |
| `text` | `str` | Raw response text |
| `response` | `OrchestratorResponse \| None` | Full response for provenance |
| `error` | `str \| None` | Set if generation failed |
| `failed` | `bool` (property) | True if error or empty text |

#### `RatingEntry`
| Field | Type | Notes |
|---|---|---|
| `label` | `str` | `"A"`, `"B"`, or `"C"` as shown to this rater |
| `score` | `float` | 1–10 |
| `reasoning` | `str` | One-sentence justification |

#### `RatingResult`
| Field | Type | Notes |
|---|---|---|
| `rater_role` | `str` | Which model rated |
| `label_to_role` | `dict[str, str]` | Maps label → model role for this rater's shuffle |
| `ratings` | `list[RatingEntry]` | Empty if rating failed |
| `raw_text` | `str` | Full model output for debugging |
| `error` | `str \| None` | Set if call or parse failed |
| `failed` | `bool` (property) | True if error or no ratings |

#### `P2QuestionResult`
| Field | Type |
|---|---|
| `case` | `EvaluationCase` — original question + ground truth |
| `answers` | `list[ModelAnswer]` — one per council role |
| `ratings` | `list[RatingResult]` — one per rater |
| `best_answer` | `ModelAnswer \| None` (property) — highest avg-scored model for this question |

`best_answer` averages every score each model received across all raters for
this question and returns the `ModelAnswer` of the highest-scoring role.

#### `DatasetVoteSummary`
| Field | Type | Notes |
|---|---|---|
| `dataset_name` | `str` | e.g. `"fever"` |
| `scores_by_role` | `dict[str, float]` | Average score per model across all questions |
| `winner` | `str` | Role with highest average |
| `question_count` | `int` | Number of questions that contributed |

#### `P2PolicyResult`
| Field | Type |
|---|---|
| `results` | `list[P2QuestionResult]` |
| `skipped_question_ids` | `list[str]` — all 3 models failed |
| `dataset_votes` | `list[DatasetVoteSummary]` — sorted by dataset name |

### Anti-bias shuffle

Each rater gets a different A/B/C shuffle so no model always sees its own
answer under the same label:

```
Rater "qa"        sees: A=qa's answer,        B=reasoning's, C=general's
Rater "reasoning" sees: A=reasoning's answer,  B=general's,   C=qa's
Rater "general"   sees: A=general's answer,    B=qa's,        C=reasoning's
```

Shuffle is deterministic (seeded) so runs are reproducible. `label_to_role`
in each `RatingResult` records the exact mapping so scores can be decoded
back to model roles.

### `cases=` parameter on `run()`

```python
# Pass questions directly — skips dataset loading
result = await policy.run(cases=my_evaluation_cases)

# Or load from profiles as usual
result = await policy.run()                         # all 5 datasets
result = await policy.run(profiles=[FeverProfile()]) # one dataset
```

This is how `p2_benchmark.py` feeds the exact same questions P1 uses to P2
for a fair comparison.

### Edge cases handled

| Scenario | Behaviour |
|---|---|
| All 3 models fail to answer | Question skipped, ID added to `skipped_question_ids` |
| 1–2 models fail | Proceeds with `[NO RESPONSE]` shown to raters |
| Rating call raises | `RatingResult.error` set, continues |
| Rating JSON malformed | Regex fallback; if both fail, error set |
| All raters fail for a question | Still in `results` but no contribution to `dataset_votes` |
| Answer too long | Truncated to 2000 chars before combining in rating prompt |
| No cases sampled | Returns empty `P2PolicyResult` immediately |

---

## 3. `run.py` — CLI entry point (`council-p2`)

Runs the standalone P2 policy and saves results to JSON.

```bash
uv run council-p2 --provider nvidia --n-per-dataset 5 --output p2_results.json
```

### CLI args

| Arg | Default | Description |
|---|---|---|
| `--provider` | `openrouter` | `openrouter` / `openai` / `huggingface` / `nvidia` |
| `--api-base` | auto | Override API base URL (auto-set for nvidia) |
| `--qa-model` | provider default | Model for qa role |
| `--reasoning-model` | provider default | Model for reasoning role |
| `--general-model` | provider default | Model for general role |
| `--n-per-dataset` | `5` | Questions per dataset |
| `--output` | `p2_results.json` | Path to save results JSON |

### Provider defaults

| Provider | qa | reasoning | general |
|---|---|---|---|
| openrouter | llama-3.1-8b-instruct | deepseek-r1 | qwen-2.5-72b-instruct |
| openai | gpt-4o-mini | o1-mini | gpt-4o |
| huggingface | Llama-3.1-8B-Instruct | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-72B-Instruct |
| nvidia | llama-3.1-8b-instruct | llama-3.1-nemotron-70b-instruct | llama-3.3-70b-instruct |

### Output JSON structure

```json
{
  "run_at": "2026-04-19T...",
  "provider": "nvidia",
  "n_per_dataset": 5,
  "total_succeeded": 25,
  "total_skipped": 0,
  "dataset_votes": [
    {"dataset_name": "fever", "winner": "qa", "scores_by_role": {"qa": 10.0, "general": 9.5}}
  ],
  "questions": [
    {
      "example_id": "...",
      "dataset_name": "fever",
      "question": "...",
      "best_model": "qa",
      "best_answer": "...",
      "answers": [{"role": "qa", "text": "..."}, ...],
      "ratings": [{"rater_role": "qa", "scores": {"qa": 10.0, "general": 9.5}}]
    }
  ]
}
```

---

## 4. `p2_benchmark.py` + `bench_cli.py` — Benchmark runner (`council-p2-bench`)

Runs P2 on the same datasets as P1 and scores with task-eval metrics for
direct comparison.

```bash
uv run council-p2-bench --provider nvidia --n-per-dataset 50
uv run council-p2-bench --provider nvidia --n-per-dataset 10 --datasets fever musique
```

### How it differs from `council-p2`

| | `council-p2` | `council-p2-bench` |
|---|---|---|
| Output | Single JSON, no metrics | Scored JSONL + summaries with task-eval metrics |
| Purpose | Quick run / demo | Fair comparison against P1 |
| Format | Council-centric | Per-question scored records |

### Output directory structure

```
p2_benchmark_results/
  p2_suite_<timestamp>/
    scores/
      musique.jsonl        ← one scored record per question
      fever.jsonl
    summaries/
      musique.json         ← council winner + aggregated metrics per dataset
      fever.json
    suite_metrics.json     ← all datasets combined
```

---

## 5. `p1_cli.py` — Individual model benchmark (`council-p1`)

Runs individual models (P1 baseline) on the same datasets as P2 for comparison.

```bash
uv run council-p1 --provider nvidia --n-per-dataset 50 --roles qa general --output-root ./p1_bench_output
uv run council-p1 --provider nvidia --n-per-dataset 5 --datasets fever --roles qa general
```

### CLI args

| Arg | Default | Description |
|---|---|---|
| `--provider` | `openrouter` | Same providers as council-p2 |
| `--roles` | `qa general` | Which roles to benchmark individually |
| `--datasets` | all 5 | Dataset names to run |
| `--n-per-dataset` | `50` | Questions per dataset |
| `--output-root` | `p1_bench_output` | Directory to save results |

### Output structure

```
p1_bench_output/
  p1_run_<timestamp>/
    scores/
      musique__qa.jsonl       ← one record per question for qa role
      musique__general.jsonl  ← one record per question for general role
    summaries/
      musique__qa.json        ← aggregated metrics for qa on musique
      musique__general.json
    suite_metrics.json        ← all datasets + roles combined
```

---

## 6. `policy_adapters.py` — `P2DatasetAdapter`

Plugs P2 blind peer-review into the unified `CouncilBenchmarkRunner` framework
so it runs alongside P3 and P4 with the same execution model.

### Architecture (from Policy Runner Notes)

```
Same PromptRequests
      ↓
CouncilBenchmarkRunner
  ├── P2DatasetAdapter   ← blind peer-review (this)
  ├── P3Adapter          ← task routing
  └── P4Adapter          ← decompose + synthesize
      ↓
All return PolicyResult (same format)
      ↓
Metric calculator on top
```

### Three phases

**Phase 1 — `plan()`**
Fans out the question to all 3 council roles, registers `SpecialistRequest`
for each. No LLM calls yet — just declares what needs to run.

**Phase 2 — `complete_specialist_phase()`**
After `CouncilBenchmarkRunner` has executed all specialist calls and cached
responses, this phase runs the rating:
- Reads all 3 answers from `SpecialistCache`
- For each rater: applies anti-bias shuffle, builds rating prompt, calls
  `execute_vote_request`
- Parses scores, computes per-role averages
- Sets `state.winner_response` to the highest-scoring model's response

**Phase 3 — `finalize()`**
Returns `make_policy_result(state)` — a `PolicyResult` with the winner's
`PromptResponse` and metadata including `scores_by_role` and `winner_role`.

### Key difference from `P2PromptAdapter`

| | `P2PromptAdapter` | `P2DatasetAdapter` |
|---|---|---|
| Selection method | Majority vote (pick one letter) | Numerical rating 1–10, avg scores |
| Rater bias mitigation | No | Anti-bias shuffle per rater |
| Use case | Single prompt voting | Dataset-level quality assessment |

---

## 7. `task_eval/profiles.py` — split fixes

`HardMathProfile` (`math-ai/MATH-500`) and `HumanEvalPlusProfile`
(`openai/openai_humaneval`) only have a `test` split, not `validation`.
The base class defaulted to `split="validation"` which caused:

```
ValueError: Bad split: validation. Available splits: ['test']
```

Fixed by adding `split: str = "test"` to both profiles.

---

## Comparing P1 vs P2

### Run P1 (individual models)

```bash
uv run council-p1 --provider nvidia --n-per-dataset 50 --roles qa general --output-root ./p1_bench_output
```

### Run P2 (council)

```bash
uv run council-p2-bench --provider nvidia --n-per-dataset 50 --output-root ./p2_bench_output
```

### Sample comparison results (n=2 per dataset, NVIDIA provider)

| Dataset | Metric | P1 qa | P1 general | P2 council | P2 winner |
|---|---|---|---|---|---|
| musique | token_f1 | 1.000 | 1.000 | 1.000 | general |
| quality | token_f1 | 0.125 | 0.667 | 0.458 | qa |
| fever | label_accuracy | 1.000 | 1.000 | 1.000 | qa |
| hardmath | math_exact_match | 0.500 | 1.000 | 1.000 | general |
| humaneval_plus | pass_at_1 | 0.000 | 0.000 | 0.000 | general |

**Observations:**
- `hardmath`: P2 council matched P1 general (1.0), beating P1 qa (0.5). Council correctly picked `general`.
- `quality`: P2 council underperformed both individual models — council picked `qa` but `general` was better.
- `humaneval_plus`: all zero because Apptainer is not available locally for code execution.

Note: n=2 per dataset is not statistically significant. Run with `--n-per-dataset 50+` for reliable results.

---

## What P2 does NOT do

- **Does not judge objective correctness.** Ratings are peer-perceived quality. Ground-truth scoring is done by `task_eval`.
- **Does not synthesize answers.** P2 rates and selects — combining is `synthesis.synthesize` used by P3/P4.
- **Does not talk to providers directly.** All calls go through `orchestrator.get_client(role).get_response(request)`.
