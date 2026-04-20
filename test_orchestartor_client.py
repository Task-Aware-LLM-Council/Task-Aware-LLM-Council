import asyncio
from llm_gateway import PromptRequest, Provider, ProviderConfig
from model_orchestration import (
    ModelOrchestrator, 
    build_default_orchestrator_config, 
    build_default_local_vllm_orchestrator_config,
    OrchestratorConfig,
    ModelSpec
)
from common import get_current_user

api_specialist_config = build_default_orchestrator_config(
    provider=Provider.OPENAI_COMPATIBLE,
    api_base="https://integrate.api.nvidia.com/v1/chat/completions",
    qa_model="google/gemma-3-27b-it",
    reasoning_model="openai/gpt-oss-120b",
    general_model="qwen/qwen2.5-coder-32b-instruct",
    api_key_env="NVIDIA_API_KEY"
)


vllm_specialist_config = build_default_local_vllm_orchestrator_config()


synthesizer_str = "synthesizer"
synthesizer_config = OrchestratorConfig(
      models=(
          ModelSpec(
              role=synthesizer_str,
              model="task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2",
              aliases=(synthesizer_str,),
              provider_config=ProviderConfig(
                  provider=Provider.LOCAL,
                  default_model="task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2",
                  default_params={
                      "local_launch_image": "vllm-openai_latest.sif",
                      "local_launch_port": 8004,
                      "local_launch_bind": f"/scratch1/{get_current_user()}/.cache",
                      "local_launch_startup_timeout_seconds": 600.0,
                      "local_launch_max_model_len": "8192",
                      "local_launch_gpu_memory_utilization": 0.50,
                      "local_launch_quantization": "compressed-tensors",
                      "local_launch_use_gpu": True,
                  },
              ),
          ),
      ),
      default_role=synthesizer_str,
      mode_label="local",
  )

async def main():
    question = "Provide me just 1 prrof of 2 lines for pythagoras theorem"
    prompt_request = PromptRequest(user_prompt=question)
    question_response = dict()
    async with ModelOrchestrator(vllm_specialist_config) as orchestrator:
        print("-----------Starting specialists-----------------")
        await orchestrator.load_all(max_parallel=1)
        print("---------All specialists started--------------")
        response = await orchestrator.qa_client.get_response(prompt_request)
        print(response.text)
        print(response.model)
        print(response.provider)
        print(response.usage.total_tokens)
        question_response["model1"] =  response.text

        response = await orchestrator.reasoning_client.get_response(prompt_request)
        print(response.text)
        print(response.model)
        print(response.provider)
        print(response.usage.total_tokens)
        question_response["model2"] = response.text

        response = await orchestrator.general_client.get_response(prompt_request)
        print(response.text)
        print(response.model)
        print(response.provider)
        print(response.usage.total_tokens)
        question_response["model3"] = response.text

        print("---------Work of Specialists is done-----------")
    
    async with ModelOrchestrator(synthesizer_config) as orchestrator:
        print("-----------Starting synthesizer-----------------")
        await orchestrator.load_all(max_parallel=1)
        print("------------Syntehsizer started----------------")
        response = await orchestrator.get_client(synthesizer_str).get_response(
            PromptRequest(user_prompt=
                          f"I asked a question {question} to 3 llm models, and they responded back with there answers"
                          f"Their answers {question_response}"
                          f"Your task is to consolidate this answer and provide who is the best model to answer the qeustion"
                          )
        )
        print(response.text)
        print(response.model)
        print(response.provider)
        print(response.usage.total_tokens)

        print("---------Synthesizer work is done-----------")

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())