"""P4: learned task-aware routing.

Joint Seq2Seq decomposer+router + council policy that drives specialist
dispatch and ordered synthesis through the CouncilBenchmarkRunner.
"""
from council_policies.p4.policy import LearnedRouterPolicy
from council_policies.p4.seq2seq_decomposer_router import (
    INPUT_PREFIX,
    GenerateFn,
    HFSeq2SeqGenerate,
    Seq2SeqDecomposerRouter,
    parse_targets,
    serialize_targets,
)

__all__ = [
    "LearnedRouterPolicy",
    "Seq2SeqDecomposerRouter",
    "HFSeq2SeqGenerate",
    "GenerateFn",
    "INPUT_PREFIX",
    "parse_targets",
    "serialize_targets",
]
