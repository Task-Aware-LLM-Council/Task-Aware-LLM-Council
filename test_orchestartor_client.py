import asyncio
import json
from datasets import load_dataset
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
                    # "local_launch_max_model_len": "8192",
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

task_aware_router_str = "task-aware-router"
task_aware_router = OrchestratorConfig(
    models=(
        ModelSpec(
            role=task_aware_router_str,
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            aliases=(task_aware_router_str,),
            provider_config=ProviderConfig(
                provider=Provider.LOCAL,
                default_model="HuggingFaceTB/SmolLM2-135M-Instruct",
                default_params={
                    "local_launch_port": 8005,
                    "local_launch_bind": f"/scratch1/{get_current_user()}/.cache",
                    "local_launch_startup_timeout_seconds": 600.0,
                    "local_launch_gpu_memory_utilization": 0.5,
                    "local_launch_use_gpu": True,
                    "local_launch_image": "vllm-openai_latest.sif",
                },
            ),
        ),
    ),
    default_role=task_aware_router_str,
    mode_label="local",
)

ROW_CONCURRENCY = 25
row_sem = asyncio.Semaphore(ROW_CONCURRENCY)


async def run_router_for_row(i, row, task_aware_orchestrator, rows):
    async with row_sem:
        question = row["question"]
        context = row.get("context", "")

        prompt = PromptRequest(
            user_prompt=f"Question: {question}\nContext: {context}\n\nAnswer the question in under 50 words."
        )

        router_client = task_aware_orchestrator.get_client(task_aware_router_str)
        router_response = await router_client.get_response(prompt)

        rows[i]["router_response"] = router_response.text


async def run_specialists_for_row(i, row, specialist_orchestrator, rows):
    async with row_sem:
        question = row["question"]
        context = row.get("context", "")

        prompt = PromptRequest(
            user_prompt=f"Question: {question}\nContext: {context}\n\nAnswer the question in under 50 words."
        )

        r1_task = specialist_orchestrator.qa_client.get_response(prompt)
        r2_task = specialist_orchestrator.reasoning_client.get_response(prompt)
        r3_task = specialist_orchestrator.general_client.get_response(prompt)

        r1, r2, r3 = await asyncio.gather(r1_task, r2_task, r3_task)

        rows[i]["model1"] = r1.text
        rows[i]["model2"] = r2.text
        rows[i]["model3"] = r3.text

    
SYNTH_CONCURRENCY = 25
synth_sem = asyncio.Semaphore(SYNTH_CONCURRENCY)

async def run_one_synth(item, synth_client):
    async with synth_sem:
        synth_prompt = PromptRequest(
            user_prompt=f"""
You are given one question and 4 model answers.

Question:
{item["question"]}

Model 1 answer:
{item["model1"]}

Model 2 answer:
{item["model2"]}

Model 3 answer:
{item["model3"]}

Model 4 answer:
{item["router_response"]}

Task:
1. Give the best final answer to the question.
2. Say which model answered best for this question.
3. Briefly explain why.
"""
        )

        synth_response = await synth_client.get_response(synth_prompt)

        return {
            **item,
            "synthesized_answer": synth_response.text
        }


async def main():
    dataset = load_dataset("task-aware-llm-council/router_dataset", split="test")

    # One shared list of dicts; every stage updates the same dict for each row.
    rows = []
    for i, row in enumerate(dataset):
        rows.append({
            "index": i,
            "question": row["question"],
            "context": row.get("context", ""),
            "gold_answer": row.get("gold_answer"),
            "gold_label": row.get("gold_label"),
        })

    async with ModelOrchestrator(task_aware_router) as task_aware_orchestrator:
        print("-----------Starting router-----------------")
        await task_aware_orchestrator.load_all(max_parallel=1)
        print("---------router started--------------")

        router_tasks = [
            run_router_for_row(i, row, task_aware_orchestrator, rows)
            for i, row in enumerate(rows)
        ]
        await asyncio.gather(*router_tasks)

    print("Router done")

    async with ModelOrchestrator(vllm_specialist_config) as specialist_orchestrator:
        print("-----------Starting specialists-----------------")
        await specialist_orchestrator.load_all(max_parallel=1)
        print("---------All specialists started--------------")

        specialist_tasks = [
            run_specialists_for_row(i, row, specialist_orchestrator, rows)
            for i, row in enumerate(rows)
        ]
        await asyncio.gather(*specialist_tasks)

    print("---------Work of Specialists is done-----------")

    final_results = []

    async with ModelOrchestrator(synthesizer_config) as synth_orchestrator:
        print("-----------Starting synthesizer-----------------")
        await synth_orchestrator.load_all(max_parallel=1)
        print("------------Syntehsizer started----------------")
        synth_client = synth_orchestrator.get_client(synthesizer_str)

        tasks = [
            run_one_synth(item, synth_client)
            for item in rows
        ]
        final_results = await asyncio.gather(*tasks)

        print("---------Synthesizer work is done-----------")
        # print(f"Final Results: {final_results}")
        output_file = "results.jsonl"

        with open(output_file, "w") as f:
            for row in final_results:
                f.write(json.dumps(row) + "\n")

        print(f"Saved results to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
































# import asyncio
# from datasets import load_dataset
# from llm_gateway import PromptRequest, Provider, ProviderConfig
# from model_orchestration import (
#     ModelOrchestrator, 
#     build_default_orchestrator_config, 
#     build_default_local_vllm_orchestrator_config,
#     OrchestratorConfig,
#     ModelSpec
# )
# from common import get_current_user

# api_specialist_config = build_default_orchestrator_config(
#     provider=Provider.OPENAI_COMPATIBLE,
#     api_base="https://integrate.api.nvidia.com/v1/chat/completions",
#     qa_model="google/gemma-3-27b-it",
#     reasoning_model="openai/gpt-oss-120b",
#     general_model="qwen/qwen2.5-coder-32b-instruct",
#     api_key_env="NVIDIA_API_KEY"
# )


# vllm_specialist_config = build_default_local_vllm_orchestrator_config()


# synthesizer_str = "synthesizer"
# synthesizer_config = OrchestratorConfig(
#       models=(
#           ModelSpec(
#               role=synthesizer_str,
#               model="task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2",
#               aliases=(synthesizer_str,),
#               provider_config=ProviderConfig(
#                   provider=Provider.LOCAL,
#                   default_model="task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2",
#                   default_params={
#                       "local_launch_image": "vllm-openai_latest.sif",
#                       "local_launch_port": 8004,
#                       "local_launch_bind": f"/scratch1/{get_current_user()}/.cache",
#                       "local_launch_startup_timeout_seconds": 600.0,
#                       "local_launch_max_model_len": "8192",
#                       "local_launch_gpu_memory_utilization": 0.50,
#                       "local_launch_quantization": "compressed-tensors",
#                       "local_launch_use_gpu": True,
#                   },
#               ),
#           ),
#       ),
#       default_role=synthesizer_str,
#       mode_label="local",
#   )


# task_aware_router_str = "task-aware-router"
# task_aware_router = OrchestratorConfig(
#       models=(
#           ModelSpec(
#               role=task_aware_router_str,
#               model="google/flan-t5-small ",
#               aliases=(task_aware_router_str,),
#               provider_config=ProviderConfig(
#                   provider=Provider.LOCAL,
#                   default_model="google/flan-t5-small ",
#                   default_params={
#                       "local_launch_image": "vllm-openai_latest.sif",
#                       "local_launch_port": 8005,
#                       "local_launch_bind": f"/scratch1/{get_current_user()}/.cache",
#                       "local_launch_startup_timeout_seconds": 600.0,
#                     #   "local_launch_max_model_len": "8192",
#                       "local_launch_gpu_memory_utilization": 0.0185,
#                     #   "local_launch_quantization": "compressed-tensors",
#                       "local_launch_use_gpu": True,
#                   },
#               ),
#           ),
#       ),
#       default_role=task_aware_router_str,
#       mode_label="local",
#   )

# async def main():
#     dataset = load_dataset("task-aware-llm-council/router_dataset", split="test")
#     # question = "Provide me just 1 prrof of 2 lines for pythagoras theorem"
#     # prompt_request = PromptRequest(user_prompt=question)
#     all_results = dict()
#     async with ModelOrchestrator(vllm_specialist_config) as specialist_orchestrator:
#         print("-----------Starting specialists-----------------")
#         await specialist_orchestrator.load_all(max_parallel=1)
#         print("---------All specialists started--------------")
#         async with ModelOrchestrator(task_aware_router) as task_aware_orchestrator:
#             # print("-----------Starting task aware router-----------------")
#             await task_aware_orchestrator.load_all(max_parallel=1)
#             # print("---------Task aware router loadedls--------------")
            
#             # response = await orchestrator.qa_client.get_response(prompt_request)
#             # print(response.text)
#             # print(response.model)
#             # print(response.provider)
#             # print(response.usage.total_tokens)
#             # question_response["model1"] =  response.text

#             # response = await orchestrator.reasoning_client.get_response(prompt_request)
#             # print(response.text)
#             # print(response.model)
#             # print(response.provider)
#             # print(response.usage.total_tokens)
#             # question_response["model2"] = response.text

#             # response = await orchestrator.general_client.get_response(prompt_request)
#             # print(response.text)
#             # print(response.model)
#             # print(response.provider)
#             # print(response.usage.total_tokens)
#             # question_response["model3"] = response.text

#             for i, row in enumerate(dataset):
#                 question = row["question"]
#                 context = row.get("context", "")

#                 prompt = PromptRequest(
#                     user_prompt=f"Question: {question}\nContext: {context}\n\nAnswer the question."
#                 )

#                 r1 = await specialist_orchestrator.qa_client.get_response(prompt)
#                 r2 = await specialist_orchestrator.reasoning_client.get_response(prompt)
#                 r3 = await specialist_orchestrator.general_client.get_response(prompt)
#                 # r4 = await task_aware_orchestrator.get_client(task_aware_router_str).get_response(prompt)

#                 all_results.append({
#                     "index": i,
#                     "question": question,
#                     "context": context,
#                     "model1": r1.text,
#                     "model2": r2.text,
#                     "model3": r3.text,
#                     # "model4": r4.text
#                 })

#                 print(f"Stored specialist responses for row {i}")

#         print("---------Work of Specialists is done-----------")
    
#     final_results = []
#     async with ModelOrchestrator(synthesizer_config) as synth_orchestrator:
#         print("-----------Starting synthesizer-----------------")
#         await synth_orchestrator.load_all(max_parallel=1)
#         print("------------Syntehsizer started----------------")
#         synth_client = synth_orchestrator.get_client(synthesizer_str)
#         # response = await orchestrator.get_client(synthesizer_str).get_response(
#         #     PromptRequest(user_prompt=
#         #                   f"I asked a question {question} to 3 llm models, and they responded back with there answers"
#         #                   f"Their answers {question_response}"
#         #                   f"Your task is to consolidate this answer and provide who is the best model to answer the qeustion"
#         #                   )
#         # )
#         # print(response.text)
#         # print(response.model)
#         # print(response.provider)
#         # print(response.usage.total_tokens)

#         for item in all_results:
#             synth_prompt = PromptRequest(
#                 user_prompt=f"""
#             You are given one question and 3 model answers.

#             Question:
#             {item["question"]}

#             Context:
#             {item["context"]}

#             Model 1 answer:
#             {item["model1"]}

#             Model 2 answer:
#             {item["model2"]}

#             Model 3 answer:
#             {item["model3"]}

#             Task:
#             1. Give the best final answer to the question.
#             2. Say which model answered best for this question.
#             3. Briefly explain why.
#             """
#                         )

#             synth_response = await synth_client.get_response(synth_prompt)

#             final_results.append({
#                 **item,
#                 "synthesized_answer": synth_response.text
#             })

#             print(f"Synthesized row {item['index']}")

#         print("---------Synthesizer work is done-----------")
#         print(f"Final Results: {final_results}")

# # Run the async function
# if __name__ == "__main__":
#     asyncio.run(main())