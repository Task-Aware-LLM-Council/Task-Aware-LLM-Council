from __future__ import annotations

import asyncio
from dataclasses import dataclass

from council_policies import (
    DatasetCouncilPolicy,
    KeywordRouter,
    LearnedRouterPolicy,
    PassthroughDecomposer,
    RuleBasedRoutingPolicy,
)
from llm_gateway import PromptRequest, Provider, ProviderConfig
from model_orchestration import (
    DEFAULT_LOCAL_VLLM_BIND,
    ModelOrchestrator,
    ModelSpec,
    OrchestratorConfig,
    build_default_local_vllm_orchestrator_config,
    vLLM_DEFAULT_GENERAL_MODEL,
    vLLM_DEFAULT_QA_MODEL,
    vLLM_DEFAULT_REASONING_MODEL,
    build_default_orchestrator_config
)


@dataclass(slots=True, frozen=True)
class PolicyResponses:
    p2_text: str
    p3_text: str
    p4_text: str


def build_p2_p3_orchestrator_config() -> OrchestratorConfig:

    return build_default_orchestrator_config(
    provider=Provider.OPENAI_COMPATIBLE,
    api_base="https://integrate.api.nvidia.com/v1/chat/completions",
    qa_model="google/gemma-3-27b-it",
    reasoning_model="openai/gpt-oss-120b",
    general_model="qwen/qwen2.5-coder-32b-instruct",
    api_key_env="NVIDIA_API_KEY"
)

    return build_default_local_vllm_orchestrator_config()


def build_p4_specialist_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        models=(
            ModelSpec(
                role="qa_reasoning",
                model=vLLM_DEFAULT_QA_MODEL,
                aliases=("qa_reasoning",),
                provider_config=_local_provider_config(
                    model=vLLM_DEFAULT_QA_MODEL,
                    port=8000,
                ),
            ),
            ModelSpec(
                role="math_code",
                model=vLLM_DEFAULT_REASONING_MODEL,
                aliases=("math_code",),
                provider_config=_local_provider_config(
                    model=vLLM_DEFAULT_REASONING_MODEL,
                    port=8001,
                ),
            ),
            ModelSpec(
                role="fact_general",
                model=vLLM_DEFAULT_GENERAL_MODEL,
                aliases=("fact_general",),
                provider_config=_local_provider_config(
                    model=vLLM_DEFAULT_GENERAL_MODEL,
                    port=8002,
                ),
            ),
        ),
        default_role="fact_general",
        mode_label="local",
    )


def build_synthesizer_config() -> OrchestratorConfig:
    synthesizer_model = vLLM_DEFAULT_REASONING_MODEL
    return OrchestratorConfig(
        models=(
            ModelSpec(
                role="synthesizer",
                model=synthesizer_model,
                aliases=("synthesizer",),
                provider_config=_local_provider_config(
                    model=synthesizer_model,
                    port=8004,
                ),
            ),
        ),
        default_role="synthesizer",
        mode_label="local",
    )


def _local_provider_config(*, model: str, port: int) -> ProviderConfig:
    return ProviderConfig(
        provider=Provider.LOCAL,
        default_model=model,
        default_params={
            "local_launch_image": "vllm-openai_latest.sif",
            "local_launch_port": port,
            "local_launch_bind": DEFAULT_LOCAL_VLLM_BIND,
            "local_launch_startup_timeout_seconds": 600.0,
            "local_launch_max_model_len": "8192",
            "local_launch_gpu_memory_utilization": 0.50,
            "local_launch_quantization": "compressed-tensors",
            "local_launch_use_gpu": True,
        },
    )


async def run_p2(question: str):
    request = PromptRequest(
        user_prompt=question,
        metadata={"example_id": "manual-p2", "dataset_name": "manual"},
    )
    async with ModelOrchestrator(build_p2_p3_orchestrator_config()) as orchestrator:
        await orchestrator.load_all(max_parallel=1)
        policy = DatasetCouncilPolicy(orchestrator)
        return await policy.run(request)


async def run_p3(question: str):
    request = PromptRequest(
        user_prompt=question,
        metadata={"example_id": "manual-p3", "dataset_name": "manual"},
    )
    async with ModelOrchestrator(build_p2_p3_orchestrator_config()) as orchestrator:
        await orchestrator.load_all(max_parallel=1)
        policy = RuleBasedRoutingPolicy(
            orchestrator,
            fallback_role="general",
            synthesizer_role="general",
        )
        return await policy.run(request)


async def run_p4(question: str):
    request = PromptRequest(
        user_prompt=question,
        metadata={"example_id": "manual-p4", "dataset_name": "manual"},
    )
    policy = LearnedRouterPolicy(
        specialist_config=build_p4_specialist_config(),
        synthesizer_config=build_synthesizer_config(),
        router=KeywordRouter(),
        decomposer=PassthroughDecomposer(),
        fallback_role="fact_general",
        synthesizer_role="synthesizer",
    )
    return await policy.run(request)


async def main() -> PolicyResponses:
    question = "Give a short proof of the Pythagorean theorem."

    p2_response = await run_p2(question)
    # p3_response = await run_p3(question)
    # p4_response = await run_p4(question)

    print("P2:", p2_response.text)
    # print("P3:", p3_response.text)
    # print("P4:", p4_response.text)

    # return PolicyResponses(
    #     p2_text=p2_response.text,
    #     # p3_text=p3_response.text,
    #     # p4_text=p4_response.text,
    # )


if __name__ == "__main__":
    asyncio.run(main())
