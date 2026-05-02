"""P4: learned task-aware routing.

Joint Seq2Seq decomposer+router + council policy that drives specialist
dispatch and ordered synthesis through the CouncilBenchmarkRunner.
"""
from council_policies.p4.decomposer import LLMDecomposer, PassthroughDecomposer
from council_policies.p4.hf_causal_generate import HFCausalGenerate
from council_policies.p4.policy import LearnedRouterPolicy
from council_policies.p4.policy_adapters import P3Adapter
from council_policies.p4.policy_runner import (
    CouncilBenchmarkRunner,
    PolicyMetrics,
    PolicyRuntime,
)
from council_policies.p4.router import (
    LearnedRouter,
    RoutingDecision,
    Subtask,
)
from council_policies.p4.router_card import DecomposerRouterCard, RouterCard
from council_policies.p4.seq2seq_decomposer_router import (
    INPUT_PREFIX,
    GenerateFn,
    HFSeq2SeqGenerate,
    Seq2SeqDecomposerRouter,
    parse_targets,
    serialize_targets,
)
from council_policies.p4.types import TASK_TO_ROLE, CouncilResponse, TaskType

__all__ = [
    "CouncilBenchmarkRunner",
    "CouncilResponse",
    "DecomposerRouterCard",
    "GenerateFn",
    "HFCausalGenerate",
    "HFSeq2SeqGenerate",
    "INPUT_PREFIX",
    "LLMDecomposer",
    "LearnedRouter",
    "LearnedRouterPolicy",
    "P3Adapter",
    "PassthroughDecomposer",
    "PolicyMetrics",
    "PolicyRuntime",
    "RouterCard",
    "RoutingDecision",
    "Seq2SeqDecomposerRouter",
    "Subtask",
    "TASK_TO_ROLE",
    "TaskType",
    "parse_targets",
    "serialize_targets",
]
