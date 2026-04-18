# council_policies2

Fixed P2/P3/P4 council policies, fully wired to the `benchmarking_pipeline`.

## What was broken in `council_policies` (and what we fixed)

| # | Bug | Fix |
|---|-----|-----|
| 1 | Package missing from UV workspace | Added to root `pyproject.toml` members |
| 2 | Policies never called by benchmark pipeline | New `PolicyClient` bridges policies → `run_benchmark` |
| 3 | Tie-break arbitrator hardcoded to `"general"` | `arbitrator_role` is now a parameter on P2 and `run_vote` |
| 4 | Unguarded fallback in P3/P4 crashes on missing role | Fallback validated at init; call site wrapped in `try/except KeyError` |
| 5 | Voter shrinkage undocumented | Documented in `voter.run_vote` docstring |
| 6 | Dead `task-eval` dependency | Removed |

## Installation

```bash
uv sync
```

Optional — P4 embedding router:

```bash
uv sync --extra p4-training
```

## Quick Start

### P3 — Rule-based routing (recommended for benchmarks)

```python
import asyncio
from model_orchestration import ModelOrchestrator, build_default_orchestrator_config
from council_policies2 import RuleBasedRoutingPolicy
from council_policies2.policy_client import PolicyClient
from benchmarking_pipeline import run_benchmark, BenchmarkRunConfig, BenchmarkDataset
from llm_gateway import Provider

config = build_default_orchestrator_config(
    provider=Provider.OPENAI_COMPATIBLE,
    api_base="https://...",
    api_key_env="MY_API_KEY",
)

async def main():
    async with ModelOrchestrator(config) as orchestrator:
        policy = RuleBasedRoutingPolicy(orchestrator)
        client = PolicyClient(policy)          # <-- drop-in for run_benchmark
        result = await run_benchmark(
            datasets,                          # Iterable[BenchmarkDataset]
            run_config,                        # BenchmarkRunConfig
            client=client,
        )

asyncio.run(main())
```

### P2 — Flat council with majority vote

```python
from council_policies2 import FlatCouncilPolicy
from council_policies2.policy_client import PolicyClient

policy = FlatCouncilPolicy(
    orchestrator,
    council_roles=("qa", "reasoning", "general"),
    arbitrator_role="general",   # used for tie-breaking
)
client = PolicyClient(policy)
```

### P4 — Embedding-based learned router

```python
from council_policies2 import LearnedRouterPolicy
from council_policies2.policy_client import PolicyClient

policy = LearnedRouterPolicy(orchestrator, encoder_model="all-MiniLM-L6-v2")
policy.load()   # downloads the encoder model on first call
client = PolicyClient(policy)
```

## How `PolicyClient` works

`run_benchmark` in `benchmarking_pipeline` accepts an optional `client: BaseLLMClient`.
`PolicyClient` implements that interface: every `generate(request)` call goes through
`policy.run(request)`, and the winning `PromptResponse` is returned.

The benchmark pipeline, scorer, and storage layers are completely unaware of the
council layer — they just see a client that returns a `PromptResponse`.

Council metadata (which policy ran, the task type, vote tally) is stored in
`PromptResponse.metadata` so it can be inspected in the recorded predictions.
