# council_policies

The `council_policies` package implements the voting and aggregation logic for the **Task-Aware LLM Council** system. It defines *how* a group of language models collaborate to produce a single best answer — rather than relying on any one model alone.

---

## What problem does this solve?

No single LLM is best at everything. A math-focused model may outperform a general model on equations but underperform on creative writing. Instead of picking one model for all tasks, the council system:

1. Sends the question to **all 3 models** simultaneously
2. Collects all 3 answers
3. Sends **all 3 answers back** to all 3 models and asks: *"Which answer is best?"*
4. Each model **votes** — without knowing which answer it produced (answers are anonymized as A, B, C)
5. The answer with the **most votes wins**
6. If there is a 3-way tie, a single **arbitrator** model makes the final call

This approach reduces individual model bias, catches hallucinations and errors that one model alone might miss, and naturally surfaces which model types perform best on which kinds of questions.

---

## Package structure

```
council_policies/
├── src/council_policies/
│   ├── __init__.py        # Public API exports
│   ├── prompts.py         # Prompt templates and builder functions
│   └── voter.py           # Full council voting logic
└── tests/
    ├── conftest.py        # sys.path setup for workspace imports
    ├── fake_client.py     # FakeClient, FailingClient, SequentialClient (no API needed)
    ├── test_prompts.py    # Tests for prompt building and variable substitution
    ├── test_voting.py     # Tests for vote parsing, tally, majority, run_council()
    └── test_domains.py    # Domain-specific tests: math, coding, logic, English + edge cases
```

---

## How the voting flow works

```
Question
   │
   ▼  (parallel)
┌──────────────────────────────────────────┐
│  Model A answers    Model B answers    Model C answers  │
└──────────────────────────────────────────┘
   │
   ▼  All 3 answers collected
   │
   ▼  (parallel — each model sees all 3 answers, labelled anonymously as A / B / C)
┌──────────────────────────────────────────┐
│  Model A votes      Model B votes      Model C votes    │
└──────────────────────────────────────────┘
   │
   ▼  Tally votes
   │
   ├── 2 or 3 votes for same answer → WINNER found ✓
   │
   └── 1-1-1 tie → Arbitrator prompt → Final winner
```

**Why anonymize answers as A/B/C?**  
If a model knows which answer it produced, it tends to vote for itself (self-preference bias). By hiding model identity and only showing answers labelled A, B, C, each model evaluates purely on quality.

---

## Prompt design

### Voter prompt (`VOTER_PROMPT_TEMPLATE`)

Sent to each of the 3 models after all answers are collected. It includes:

- The original question
- All 3 answers labelled as **ANSWER A**, **ANSWER B**, **ANSWER C**
- Domain-specific evaluation rules (math → check steps, coding → check runnability, etc.)
- Strict output format:
  ```
  Vote: <A / B / C>
  Confidence: <High / Medium / Low>
  Reason: <explanation>
  ```

### Aggregator prompt (`AGGREGATOR_PROMPT_TEMPLATE`)

Used **only** when votes are split 1-1-1. A single arbitrator model receives:

- The question and all 3 answers
- A summary of all votes cast (with confidence levels)
- Instructions to produce a final answer, correcting any mistakes from all models if needed

Output format:
```
Winner: <A / B / C / NONE>
Confidence: <High / Medium / Low>
Reason: <explanation>
Final Answer: <best possible answer>
```

---

## Supported domains

The council handles all question types. Domain-specific evaluation rules are embedded in the voter prompt:

| Domain | Evaluation priority |
|--------|-------------------|
| **Math** | Correct final answer + valid intermediate steps |
| **Coding** | Code must be correct, runnable, and efficient |
| **Logic** | Valid deductive reasoning from given premises |
| **English / Writing** | Clarity, grammar, coherence, and factual accuracy |

---

## Public API

```python
from council_policies import run_council, ModelConfig, CouncilResult
from llm_gateway.models import ProviderConfig, Provider

# Define 3 models
models = [
    ModelConfig(
        model_id="openai/gpt-4o-mini",
        provider_config=ProviderConfig(provider=Provider.OPENROUTER, api_key_env="OPENROUTER_API_KEY"),
    ),
    ModelConfig(
        model_id="qwen/qwen3-32b",
        provider_config=ProviderConfig(provider=Provider.OPENROUTER, api_key_env="OPENROUTER_API_KEY"),
    ),
    ModelConfig(
        model_id="meta-llama/llama-3.3-70b-instruct",
        provider_config=ProviderConfig(provider=Provider.OPENROUTER, api_key_env="OPENROUTER_API_KEY"),
    ),
]

# Run the council
result: CouncilResult = await run_council(models, question="What is the integral of x^2?")

print(result.winning_model_id)   # e.g. "openai/gpt-4o-mini"
print(result.winning_answer)     # the best answer text
print(result.winner_label)       # "A", "B", or "C"
print(result.tiebreak_used)      # True if arbitrator was needed
print(result.votes)              # list of Vote objects from all 3 models
```

### Key data classes

| Class | Description |
|-------|-------------|
| `ModelConfig` | Pairs a `model_id` string with a `ProviderConfig` from `llm_gateway` |
| `CouncilAnswer` | One model's raw answer to the question |
| `Vote` | One model's vote: `voted_for` (A/B/C), `confidence`, `reason` |
| `CouncilResult` | Full result: answers, votes, winner, tiebreak flag |

### Builder functions

```python
from council_policies import build_voter_prompt, build_aggregator_prompt

# Build prompts manually (useful for debugging or custom orchestration)
voter_prompt = build_voter_prompt(
    question="What is 2 + 2?",
    answer_a="4",
    answer_b="5",
    answer_c="22",
)

aggregator_prompt = build_aggregator_prompt(
    question="What is 2 + 2?",
    answer_a="4", answer_b="5", answer_c="22",
    votes_summary="Model A voted: A\nModel B voted: A\nModel C voted: B",
)
```

---

## Running tests

No API keys required — all tests use mocked clients.

```bash
# From the workspace root
uv run pytest packages/council_policies/tests -q

# Verbose output
uv run pytest packages/council_policies/tests -v

# Specific test file
uv run pytest packages/council_policies/tests/test_domains.py -v

# Run only math domain tests
uv run pytest packages/council_policies/tests/test_domains.py::TestMathDomain -v
```

---

## Dependencies

- `llm-gateway` (workspace) — provider abstraction, `create_client()`, `PromptRequest`, `ProviderConfig`
- `task-eval` (workspace) — dataset profiles (used in future P3/P4 routing)
- Python ≥ 3.12
