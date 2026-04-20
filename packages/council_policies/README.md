# council_policies

P2/P3/P4 orchestration policies for the Task-Aware LLM Council. This package
decides *how* the council turns a user prompt into a final answer — flat
voting, rule-based routing, or learned routing — on top of the role clients
provided by `model-orchestration`.

## What lives here

| File | Purpose | Status |
|---|---|---|
| `models.py` | `CouncilResponse`, `TaskType`, `TASK_TO_ROLE` | stable |
| `prompts.py` | System prompts + prompt builders for voter/synthesizer | stable |
| `voter.py` | `run_vote()` — LLM-as-judge selection, used by P2 | stable |
| `synthesis.py` | `synthesize()` — specialist fusion, used by P3/P4 multi-skill | stable |
| `p3_policy.py` | `RuleBasedRoutingPolicy` — single-specialist routing (P3) | stable (single-skill only; multi-skill upgrade pending) |
| `p2_policy.py` | `FlatCouncilPolicy` | **not yet written** |
| `p4_policy.py` | `LearnedRouterPolicy` | **not yet written** |

## Two aggregation paradigms — which to use when

This is the single most important distinction in the package. Pick the right
one for the policy you're writing.

### 1. Voting (`voter.run_vote`) — for P2

- **Shape of input:** N candidates that each attempted the **same** task.
- **What it does:** a panel of "voter" roles reads the candidates and picks
  one. Ties are broken by a tie-break arbitrator LLM call.
- **Output:** a single `OrchestratorResponse` from the winning candidate
  (no new text is generated).
- **Use when:** every LLM saw the exact same prompt. P2's flat council is
  the canonical example.

```python
from council_policies import run_vote

winner_key, tally = await run_vote(
    question=request.user_prompt,
    candidates={"qa": resp_qa, "reasoning": resp_reason, "general": resp_gen},
    orchestrator=orchestrator,
    voter_roles=("qa", "reasoning", "general"),
)
```

### 2. Synthesis (`synthesis.synthesize`) — for P3/P4 multi-skill

- **Shape of input:** N candidates that each solved a **different slice** of
  the task (math did arithmetic, qa did passage lookup, etc.).
- **What it does:** one LLM call fuses the partials into a single coherent
  answer that preserves every claim.
- **Output:** a `SynthesisResult` wrapping a **newly generated**
  `OrchestratorResponse` (not one of the inputs).
- **Use when:** your policy decomposed the task and dispatched different
  sub-problems to different specialists.

```python
from council_policies import synthesize

result = await synthesize(
    question=request.user_prompt,
    partials={"math": resp_math, "qa": resp_qa},   # role -> OrchestratorResponse
    orchestrator=orchestrator,
    synthesizer_role="general",                    # the fuser
)
final_text = result.text
```

Single-specialist short-circuit: if `partials` has one entry, `synthesize`
returns it verbatim without an LLM call — so you can pass the dict
unconditionally from P3 even when only one specialist ran.

### Rule of thumb

> Same prompt, many tries → **vote**.
> Different sub-prompts, complementary outputs → **synthesize**.

## Writing a new policy (P2 or P4)

Each policy class should follow the shape already set by
`RuleBasedRoutingPolicy` in `p3_policy.py`:

1. `__init__(self, orchestrator: ModelOrchestrator, ...)` — never construct
   `llm_gateway` clients directly. Go through `orchestrator.get_client(role)`.
2. `async def run(self, request: PromptRequest, ...) -> CouncilResponse` —
   the single public entry point.
3. Inside `run`:
   - **(P2)** Fan the same `request` out to all configured roles in
     parallel (`asyncio.gather`). Pass the resulting dict to `run_vote`.
     Package the winner into a `CouncilResponse(policy="p2", ...)`.
   - **(P4)** Call the learned router to pick role(s). If one role, return
     its response directly (like P3). If many roles, dispatch in parallel
     and pass the partials to `synthesize`. Package as
     `CouncilResponse(policy="p4", ...)`.
4. Populate `CouncilResponse.metadata` with anything a downstream analyst
   would want to reproduce the decision (routed roles, vote tally,
   synthesizer role, router confidence, fallbacks used, …).

### What a policy should *not* do

- **Don't load datasets or score.** That's `task_eval`'s job.
- **Don't talk to providers directly.** Go through `orchestrator.get_client`.
- **Don't run answer extraction inside the policy.** If you need the
  extracted answer (e.g. for voting over normalized labels), ask the caller
  to pass it in. Extraction belongs to `task_eval.profiles`.
- **Don't hand-roll a voting or synthesis prompt.** Use `voter.run_vote` or
  `synthesis.synthesize`. Both take their system prompts from `prompts.py`
  so wording stays consistent across the package.

## Prompts (`prompts.py`)

Two paradigms, grouped in the file by section header:

| Symbol | Used by | What it's for |
|---|---|---|
| `VOTER_SYSTEM_PROMPT` | `voter.run_vote` | Each voter LLM's instruction |
| `TIEBREAK_SYSTEM_PROMPT` | `voter._break_tie` | Tie-break arbitrator's instruction |
| `build_voter_prompt` | `voter.run_vote` | User-prompt formatter for a voter call |
| `build_tiebreak_prompt` | `voter._break_tie` | User-prompt formatter for a tie-break |
| `parse_vote` | `voter.*` | Extract a single letter label from a vote response |
| `SYNTHESIZER_SYSTEM_PROMPT` | `synthesis.synthesize` | Fuser's instruction — preserves every partial's claim |
| `build_synthesis_prompt` | `synthesis.synthesize` | Labels each partial with its role |

If you need a new prompt, add it here — never inline in a policy. Keeps
A/B'ing wordings and auditing behavior trivial.

## Running tests

```bash
uv run pytest packages/council_policies/tests -q
```

There is currently no test folder populated; P2 and P4 tests land alongside
those policies. `test_synthesis.py` is scaffolded-but-skipped (contract-only)
until you wire an orchestrator stub for it.

## Dependencies (enforced by layer contract)

`council_policies` depends on `model-orchestration`, `llm_gateway`, and
`task_eval`. It does **not** depend on `benchmark_runner` or
`benchmarking_pipeline`. If you reach for one of those, you're probably
solving the wrong problem — the caller owns benchmark orchestration, not
the policy.

## Design notes and rationale

- **Two aggregators, not one.** The initial spec imagined a single
  `aggregation.py` doing deterministic selection. It was deleted in favor of
  the split above: voting is the right primitive for P2 (redundant
  attempts), synthesis is the right primitive for P3/P4 multi-skill
  (complementary attempts). Collapsing them into one file conflated two
  different problems.
- **Role vs. dataset as dispatch key.** Policies dispatch on `TaskType`
  (orchestrator role). Dataset-specific behavior (extractors, metrics)
  lives in `task_eval.profiles` and is accessed by the caller — not by
  the policy. Keep that wall clean.
- **P3 currently single-specialist.** `RuleBasedRoutingPolicy.run` picks one
  role via `classify_task`. Upgrading to multi-skill requires
  `classify_task` to return a **set** of `TaskType`s (or a threshold-ranked
  list) and `run` to parallel-dispatch, then call `synthesize`. Defer this
  until a multi-skill benchmark dataset actually exists to exercise it.
