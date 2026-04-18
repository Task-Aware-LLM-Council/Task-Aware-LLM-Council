import asyncio
from llm_gateway import PromptRequest, Provider
from model_orchestration import ModelOrchestrator, build_default_orchestrator_config

config = build_default_orchestrator_config(
    provider=Provider.OPENAI_COMPATIBLE,
    api_base="https://integrate.api.nvidia.com/v1/chat/completions",
    qa_model="google/gemma-3-27b-it",
    reasoning_model="openai/gpt-oss-120b",
    general_model="qwen/qwen2.5-coder-32b-instruct",
    api_key_env="NVIDIA_API_KEY"
)



async def main():
    async with ModelOrchestrator(config) as orchestrator:
        response = await orchestrator.qa_client.get_response(
            PromptRequest(user_prompt="Answer briefly: What is the capital of France?")
        )
        print(response.text)
        print(response.model)
        print(response.provider)
        print(response.usage.total_tokens)

        response = await orchestrator.reasoning_client.get_response(
            PromptRequest(user_prompt="Provide me just 1 proof of 2 lines for pythagoras theorem")
        )
        print(response.text)
        print(response.model)
        print(response.provider)
        print(response.usage.total_tokens)



        response = await orchestrator.general_client.get_response(
            PromptRequest(user_prompt="How's life?")
        )
        print(response.text)
        print(response.model)
        print(response.provider)
        print(response.usage.total_tokens)

if __name__ == "__main__":
    asyncio.run(main())