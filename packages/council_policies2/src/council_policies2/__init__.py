"""
council_policies2 — Fixed P2/P3/P4 policies, fully wired to the benchmark pipeline.

Bugs fixed versus council_policies
------------------------------------
1. Package is now registered as a workspace member.
2. PolicyClient bridges policies → benchmarking_pipeline.run_benchmark().
3. Tie-break arbitrator role is now configurable, not hardcoded to "general".
4. Fallback role in P3/P4 is validated at construction and guarded at call site.
5. Voter shrinkage behaviour is documented in voter.run_vote().
6. Dead task-eval dependency removed.

Quick start (P3 rule-based routing)
-------------------------------------
    from model_orchestration import ModelOrchestrator, build_default_orchestrator_config
    from council_policies2 import RuleBasedRoutingPolicy
    from council_policies2.policy_client import PolicyClient
    from benchmarking_pipeline import run_benchmark, BenchmarkRunConfig, BenchmarkDataset

    config = build_default_orchestrator_config(provider="openai-compatible", ...)
    async with ModelOrchestrator(config) as orchestrator:
        policy = RuleBasedRoutingPolicy(orchestrator)
        client = PolicyClient(policy)
        result = await run_benchmark(datasets, run_config, client=client)
"""

from council_policies2.models import TASK_TO_ROLE, CouncilResponse, TaskType
from council_policies2.p2_policy import FlatCouncilPolicy
from council_policies2.p3_policy import RuleBasedRoutingPolicy, classify_task
from council_policies2.p4_policy import LearnedRouterPolicy
from council_policies2.policy_client import PolicyClient

__all__ = [
    "CouncilResponse",
    "FlatCouncilPolicy",
    "LearnedRouterPolicy",
    "PolicyClient",
    "RuleBasedRoutingPolicy",
    "TASK_TO_ROLE",
    "TaskType",
    "classify_task",
]
