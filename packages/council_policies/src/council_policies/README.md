# council_policies — Source Reference

This document describes every module inside `src/council_policies/` and explains the decisions behind their design.

---

## `prompts.py` — Prompt templates

Contains two prompt templates and their corresponding builder functions. These are plain strings with `{placeholder}` variables — no external templating library needed.

### `VOTER_PROMPT_TEMPLATE`

The core voting prompt. It is sent to **each of the 3 council models** after all answers have been collected.

**Key design decisions:**

- **Answers are labelled A, B, C** — never by model name. This is the primary anti-bias mechanism. A model that knows which answer it produced will tend to vote for itself, even if another answer is better. Anonymization forces evaluation on content alone.
- **Temperature is set to 0.0** when sending this prompt — we want deterministic, consistent votes, not creative re-ranking.
- **Domain rules are embedded directly** in the prompt (not passed as a parameter) so the model has all context in one place. The model self-selects which rules apply based on the question type.
- **Output format is strict** — `Vote: <A/B/C>` on its own line makes regex parsing reliable even when models add extra prose.

### `AGGREGATOR_PROMPT_TEMPLATE`

The tiebreaker prompt. Used only when votes split 1-1-1 (no majority). A single arbitrator model sees:
- All 3 answers
- All 3 votes with confidence levels
- Instructions to override votes if it identifies a clear error

The aggregator can return `NONE` if all answers are wrong and it cannot determine a best one.

### Builder functions

```python
build_voter_prompt(question, answer_a, answer_b, answer_c) -> str
build_aggregator_prompt(question, answer_a, answer_b, answer_c, votes_summary) -> str
```

These are thin wrappers around `.format()` — they exist so callers never touch the raw template strings directly, making future template changes backward-compatible.

---

## `voter.py` — Council voting logic

The main orchestration module. Implements the full council flow as async functions that compose cleanly.

### Data classes

#### `ModelConfig`
```python
@dataclass
class ModelConfig:
    model_id: str            # e.g. "openai/gpt-4o-mini"
    provider_config: ProviderConfig
```
Pairs a model identifier with its provider configuration from `llm_gateway`. This is the unit you pass in when constructing a council.

#### `CouncilAnswer`
```python
@dataclass
class CouncilAnswer:
    model_config: ModelConfig
    answer: str
```
One model's raw answer to the original question. The `model_config` is retained so the winner can be identified by model ID at the end.

#### `Vote`
```python
@dataclass
class Vote:
    voter_model_id: str   # which model cast this vote
    voted_for: str        # "A", "B", or "C"
    confidence: str       # "High", "Medium", or "Low"
    reason: str           # the model's explanation
```
Parsed from the model's raw text output. The `reason` field is useful for debugging why the council made a particular decision.

#### `CouncilResult`
```python
@dataclass
class CouncilResult:
    question: str
    answers: list[CouncilAnswer]    # always 3, index 0=A, 1=B, 2=C
    votes: list[Vote]               # always 3
    winner_label: str               # "A", "B", "C", or "NONE"
    winning_answer: str
    winning_model_id: str
    tiebreak_used: bool
```
The complete output of a council run. Contains all intermediate data (answers and votes) so callers can audit the full reasoning chain, not just the final answer.

---

### Functions

#### `gather_answers(models, question) -> list[CouncilAnswer]`
Sends the question to all 3 models **in parallel** using `asyncio.gather()`. Returns answers in the same order as the input `models` list — this order determines the A/B/C mapping.

#### `gather_votes(models, question, answers) -> list[Vote]`
Builds the voter prompt with the 3 answers and sends it to all 3 models **in parallel**. Each model returns a vote. The vote is parsed from raw text by `_parse_vote()`.

#### `_parse_vote(raw_text, voter_model_id) -> Vote`
Uses regex to extract `Vote:`, `Confidence:`, and `Reason:` from model output. Designed to be resilient:
- Case-insensitive matching
- Defaults to `"A"` if no valid vote label found (graceful degradation)
- Defaults to `"Low"` confidence if field is missing
- Captures multiline reason text

#### `tally_votes(votes) -> dict[str, int]`
Counts votes per label. Returns `{"A": n, "B": n, "C": n}`.

#### `majority_winner(tally) -> str | None`
Returns the winning label if any answer has ≥ 2 votes. Returns `None` if it's a 1-1-1 tie.

#### `run_aggregator(arbitrator, question, answers, votes) -> str`
Called only on a tie. Sends the aggregator prompt to a single arbitrator model and parses its `Winner:` field. Returns the winning label (`"A"`, `"B"`, `"C"`, or `"NONE"`).

#### `run_council(models, question, arbitrator=None) -> CouncilResult`
The main entry point. Composes all of the above:
1. `gather_answers()` → 3 answers
2. `gather_votes()` → 3 votes
3. `tally_votes()` + `majority_winner()` → winner or tie
4. `run_aggregator()` if tie (defaults arbitrator to `models[0]`)
5. Returns `CouncilResult`

**Raises `ValueError`** if `len(models) != 3` — the council is designed for exactly 3 models.

---

## `__init__.py` — Public API

Exports the symbols callers need without exposing internal helpers:

```python
# Prompt builders
build_voter_prompt
build_aggregator_prompt
VOTER_PROMPT_TEMPLATE
AGGREGATOR_PROMPT_TEMPLATE

# Voter types and entry point
ModelConfig
CouncilAnswer
Vote
CouncilResult
run_council
```

Internal functions (`_parse_vote`, `_cast_vote`, `_parse_aggregator_winner`, etc.) are intentionally not exported — they are implementation details that can change without breaking callers.
