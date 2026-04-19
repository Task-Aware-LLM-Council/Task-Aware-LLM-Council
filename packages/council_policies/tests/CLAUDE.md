# Testing P3 (`RuleBasedRoutingPolicy`)

Guide for writing and maintaining tests for `council_policies/p3_policy.py`.
Read this before adding or modifying tests in this directory.

---

## What P3 does (for test intent)

P3 is three things in one class:

1. **A classifier** — `classify_task(prompt)` maps free-text → `TaskType` via
   regex keyword scoring. Falls back to `QA` when no keywords match.
2. **A dispatcher** — `run()` asks the orchestrator for the classified role's
   client and forwards the `PromptRequest`. Falls back to `fallback_role`
   when the primary role is missing.
3. **A synthesis wrapper** — when `synthesis.py` is importable, the specialist
   response is passed through `synthesize()`, which short-circuits on a
   single partial but normalizes the output shape.

Every test should target exactly one of these three concerns. Mixing them
produces tests that break for reasons unrelated to what they claim to check.

---

## The `FakeOrchestrator` pattern

Do not spin up a real `ModelOrchestrator` for unit tests — real providers
mean real API keys and real latency. Follow the `FakeClient` pattern used in
`packages/model-orchestration/tests/test_orchestrator.py:37`:

```python
class FakeClient:
    def __init__(self, role: str, *, text: str = "", fail: bool = False):
        self.role = role
        self.text = text or f"response-from-{role}"
        self.fail = fail
        self.requests: list[PromptRequest] = []

    async def get_response(self, request: PromptRequest) -> OrchestratorResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError(f"boom:{self.role}")
        return OrchestratorResponse(text=self.text, ...)  # fill per actual shape


class FakeOrchestrator:
    def __init__(self, clients: dict[str, FakeClient]):
        self._clients = clients

    def get_client(self, role: str) -> FakeClient:
        if role not in self._clients:
            raise KeyError(role)
        return self._clients[role]
```

`FakeOrchestrator` only needs `get_client(role)` — that's the entire
surface P3 touches. Do not mock more than you need.

---

## The test matrix

Organize tests into these groups. Each row is one `test_*` function.

### `test_classify.py` — classifier only

| Input                                              | Expected `TaskType` |
| -------------------------------------------------- | ------------------- |
| `"Solve the equation x^2 + 3x = 0"`                | `MATH`              |
| `"Write a Python function that sorts a list"`      | `CODE`              |
| `"Compare these two arguments step by step"`       | `REASONING`         |
| `"Is the following claim true: Paris is in Italy"` | `FEVER`             |
| `"Tell me about elephants"`                        | `QA`                |
| `""` (empty)                                       | `QA`                |
| Text matching two categories equally               | First by dict order |

Last row is a **tie-break documentation test**, not a correctness claim.
The behavior is an artifact of `max(scores.items(), key=...)` in
`classify_task()` (p3_policy.py:52). If someone deliberately changes
tie-break logic, this test will fail and force them to acknowledge it.

### `test_construction.py` — `__init__` validation

- `fallback_role="general"` + orchestrator registers `"general"` → OK.
- `fallback_role="nope"` + orchestrator has no `"nope"` → raises
  `ValueError` mentioning the bad role name. This is the fix from commit
  053ce82; regression of it silently breaks prod wiring.
- Default `fallback_role` is `"general"` (assert explicitly so nobody
  changes the default without intent).

### `test_dispatch.py` — `run()` happy path + fallback

- **Happy path:** orchestrator has all 6 role clients (`qa`, `math`,
  `code`, `reasoning`, `fever`, `general`); each task type routes to the
  expected client. Assert the corresponding `FakeClient.requests` list
  has exactly one entry.
- **Fallback fires:** orchestrator has `"general"` only; a math prompt
  routes to `general` (the fallback). Assert a warning was logged.
- **Double miss:** orchestrator has nothing. `run()` raises `RuntimeError`
  mentioning both the primary *and* fallback role names. Again a
  regression check for commit 053ce82.
- **task_type override:** passing `task_type=TaskType.CODE` bypasses
  `classify()` — even a math-sounding prompt routes to `code`. Prevents
  future "helpful" refactors from classifying the override away.

### `test_synthesis_wiring.py` — the synthesis path

Tricky because it uses a module-level `try/except ImportError` at
p3_policy.py:71. Three scenarios to cover:

- **Synthesis available, single partial:** synthesize short-circuits,
  returns verbatim. Assert `CouncilResponse.metadata["synthesis_short_circuit"]`
  is `"single_specialist"` (after the peer fixes p3_policy.py:207).
  Assert the synthesizer client was **never called** — `FakeClient.requests`
  for `synthesizer_role` is empty.
- **Synthesis raises mid-call:** monkeypatch `_synthesize` to raise.
  Policy logs a warning and returns the direct specialist response.
  Metadata carries `synthesis_available=True` but the fused path was skipped.
- **Synthesis unavailable:** set `_SYNTHESIS_AVAILABLE = False` (or mock
  the import). Policy takes the direct path. Metadata carries
  `synthesis_available=False`.

Use `monkeypatch.setattr("council_policies.p3_policy._synthesize", ...)`
and `monkeypatch.setattr("council_policies.p3_policy._SYNTHESIS_AVAILABLE", ...)`
rather than reimporting. Cleaner, no module reload games.

---

## What NOT to test

- **Don't test that `ModelOrchestrator.get_client` returns a real client.**
  That's orchestration's problem, covered in its own test suite.
- **Don't test specific regex keyword lists.** The classifier is heuristic;
  tying tests to exact keywords makes it impossible to tune without touching
  tests. Test the *categories* with representative inputs instead.
- **Don't test synthesis internals from here.** If a test needs to know what
  `SynthesisResult.metadata["short_circuit"]` equals, that's a synthesis
  unit test, not a P3 test. P3 tests should treat `synthesize()` as a
  black box mocked at the import site.
- **Don't hit real LLMs.** No network calls. Ever. If you need "does
  classifier X work on real model outputs," that's an eval, not a test.

---

## Running the tests

```bash
# Run just P3 tests
pytest packages/council_policies/tests/ -v

# Run one group
pytest packages/council_policies/tests/test_dispatch.py -v

# With coverage on the module
pytest packages/council_policies/tests/ \
  --cov=council_policies.p3_policy --cov-report=term-missing
```

Coverage target for p3_policy.py: **≥ 95%**. The module is small
(~230 lines) and every branch matters — the fallback-of-fallback path
(p3_policy.py:174-179) is exactly the kind of code that silently rots
without coverage.

---

## When you add a new policy (P1/P2/P4)

Copy this file as a starting template but retarget it. P2's voter path
has its own test matrix (tally ties, role quorum, voter role failure);
P4's decompose + multi-synthesis path is another thing entirely.
Don't try to make one CLAUDE.md cover all policies — each has its own
invariants.
