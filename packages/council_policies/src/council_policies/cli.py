# from __future__ import annotations

# import argparse
# import asyncio
# import json
# import logging
# from dataclasses import replace
# from pathlib import Path

# from llm_gateway import Provider
# from model_orchestration import ModelOrchestrator, build_default_orchestrator_config
# from benchmarking_pipeline import BenchmarkRunConfig, run_benchmark

# # We import the core suite logic from benchmark_runner
# from benchmark_runner.config import default_provider_config, get_dataset_configs, get_preset_spec
# from benchmark_runner.suite import run_registered_benchmark_suite
# from common import get_current_user

# from council_policies.adapter import PolicyClient
# from council_policies.p3_policy import RuleBasedRoutingPolicy

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # Default free/open models for the specialist backend.
# # ---------------------------------------------------------------------------
# _DEFAULT_QA_MODEL = "google/gemma-2-9b-it:free"
# _DEFAULT_REASONING_MODEL = "mistralai/mistral-7b-instruct:free"
# _DEFAULT_GENERAL_MODEL = "meta-llama/llama-3.1-8b-instruct:free"


# def build_parser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(description="Run the P3 RuleBasedRoutingPolicy via Benchmark Runner.")
#     p.add_argument("--preset", default="pilot", choices=("pilot", "full"))
#     p.add_argument(
#         "--output-root",
#         default=f"/scratch1/{get_current_user()}/results/benchmark_suite",
#         help="Directory that will contain the suite output directory.",
#     )
#     p.add_argument(
#         "--datasets",
#         nargs="*",
#         help="Optional dataset subset. Defaults to all configured datasets.",
#     )
#     p.add_argument(
#         "--sample-cap",
#         type=int,
#         help="Optional override for max examples per dataset.",
#     )
#     p.add_argument(
#         "--split",
#         help="Optional split override applied to all selected datasets.",
#     )
#     p.add_argument(
#         "--provider",
#         default="openrouter",
#         choices=("openrouter", "openai", "openai-compatible", "local", "huggingface"),
#         help="LLM provider for backend specialists. Default: openrouter",
#     )
#     p.add_argument("--api-base", default=None, help="Provider API base URL override.")
#     p.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="Env var name holding the API key.")
#     p.add_argument("--qa-model", default=_DEFAULT_QA_MODEL, help="Model for QA tasks.")
#     p.add_argument("--reasoning-model", default=_DEFAULT_REASONING_MODEL, help="Model for reasoning/math/code tasks.")
#     p.add_argument("--general-model", default=_DEFAULT_GENERAL_MODEL, help="Model for general/FEVER tasks (fallback).")
#     p.add_argument("--fallback-role", default="general", help="Role used when specialist is not registered.")
#     p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
#     return p


# async def run_cli_async(args: argparse.Namespace) -> int:
#     if args.verbose:
#         logging.basicConfig(level=logging.DEBUG)
#     else:
#         logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

#     # 1. Prepare benchmark spec configuration
#     output_root = Path(args.output_root)
#     spec = get_preset_spec(args.preset, output_root=output_root)

#     # Force the model array in spec to exactly one virtual "model"
#     # because the pipeline is evaluating the policy as a whole.
#     virtual_model_name = "p3-router"
#     spec = replace(spec, models=(virtual_model_name,))

#     if args.sample_cap is not None:
#         spec = replace(spec, max_examples_per_dataset=args.sample_cap)

#     dataset_configs = get_dataset_configs(tuple(args.datasets) if args.datasets else None)
#     if args.split:
#         dataset_configs = tuple(
#             type(config)(name=config.name, split=args.split, profile_kwargs=dict(config.profile_kwargs))
#             for config in dataset_configs
#         )

#     # 2. Setup the backend model orchestration for P3 specialists
#     provider_map = {
#         "openrouter": Provider.OPENROUTER,
#         "openai": Provider.OPENAI,
#         "openai-compatible": Provider.OPENAI_COMPATIBLE,
#         "local": Provider.LOCAL,
#         "huggingface": Provider.HUGGINGFACE,
#     }
#     backend_provider = provider_map[args.provider]

#     backend_config = build_default_orchestrator_config(
#         provider=backend_provider,
#         api_base=args.api_base,
#         api_key_env=args.api_key_env,
#         qa_model=args.qa_model,
#         reasoning_model=args.reasoning_model,
#         general_model=args.general_model,
#     )

#     # 3. Enter orchestrator and inject the wrapped Policy Client into benchmark suite
#     async with ModelOrchestrator(backend_config) as orchestrator:
#         policy = RuleBasedRoutingPolicy(
#             orchestrator,
#             fallback_role=args.fallback_role,
#         )
#         policy_client = PolicyClient(policy, model_name=virtual_model_name)

#         # Custom runner factory bridging benchmark runner and our instantiated client
#         async def custom_pipeline_runner(datasets_iter, pipeline_config: BenchmarkRunConfig):
#             # We must pass the exact datasets iterable and config downwards
#             return await run_benchmark(
#                 datasets_iter,
#                 pipeline_config,
#                 client=policy_client  # <--- Inject adapter
#             )

#         logger.info("Starting P3 integration via Benchmark Runner suite...")
        
#         result = await run_registered_benchmark_suite(
#             dataset_configs,
#             spec,
#             pipeline_runner=custom_pipeline_runner,
#         )

#     # 4. Output the standard suite summary
#     payload = {
#         "suite_id": result.suite_id,
#         "output_dir": str(result.output_dir),
#         "aggregate_summary_path": str(result.aggregate_summary_path),
#         "total_pairs": result.total_pairs,
#         "total_examples": result.total_examples,
#         "scored_examples": result.scored_examples,
#         "failed_examples": result.failed_examples,
#     }
#     print(json.dumps(payload, indent=2, sort_keys=True))
#     return 0


# def main(argv: list[str] | None = None) -> int:
#     parser = build_parser()
#     args = parser.parse_args(argv)
#     return asyncio.run(run_cli_async(args))
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

from typing import List
from task_eval.models import EvaluationCase

from benchmarking_pipeline import BenchmarkExample
from council_policies.adapter import PolicyClient
from council_policies.p3_policy import RuleBasedRoutingPolicy

from common import get_current_user

api_specialist_config = build_default_orchestrator_config(
    provider=Provider.OPENAI_COMPATIBLE,
    api_base="https://integrate.api.nvidia.com/v1/chat/completions",
    qa_model="google/gemma-3-27b-it",
    reasoning_model="openai/gpt-oss-120b",
    general_model="qwen/qwen2.5-coder-32b-instruct",
    api_key_env="NVIDIA_API_KEY"
)

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

async def main():
    dataset = load_dataset("task-aware-llm-council/router_dataset", split="test")

    # One shared list of dicts; every stage updates the same dict for each row.
    rows : List[EvaluationCase] = []
    for i, row in enumerate(dataset):
        case = EvaluationCase(
            example=BenchmarkExample(
                question=row["question"],
                context=row.get("context", ""),
                # gold_answer=row.get("gold_answer"),
                # gold_label=row.get("gold_label") , 
                example_id=row.get("example_id", str(i)),
                dataset_name = row.get("source_dataset" , "")
                # system_prompt: str | None = None
                # messages: tuple[Message, ...] = ()
                # metadata: dict[str, Any] = field(default_factory=dict)

            )
        )

        rows.append(case)

    result = None
    async with ModelOrchestrator(api_specialist_config) as specialist_orchestrator:
        print("-----------Starting specialists-----------------")
        policy = RuleBasedRoutingPolicy(
            specialist_orchestrator,
            #fallback_role=args.fallback_role,
        )
        result = await policy.run(rows) #Extract whatever needed for policy
        #     "gold_label": row.get("gold_label"),
        # })

    # async with ModelOrchestrator(api_specialist_config) as specialist_orchestrator:
    #     print("-----------Starting specialists-----------------")
    #     policy = RuleBasedRoutingPolicy(
    #         specialist_orchestrator,
    #         #fallback_role=args.fallback_role,
    #     )
    #     policy.run(rows) #Extract whatever needed for policy

    #     #await specialist_orchestrator.load_all(max_parallel=1)
    #     print("---------All specialists started--------------")

    #     specialist_tasks = [
    #         run_specialists_for_row(i, row, specialist_orchestrator, rows)
    #         for i, row in enumerate(rows)
    #     ]
    #     await asyncio.gather(*specialist_tasks)

    print("---------Work of Specialists is done-----------")

    final_results = [] 

    print("---------Synthesizer work is done-----------")
    # print(f"Final Results: {final_results}")
    output_file = "results.jsonl"

    with open(output_file, "w") as f:
        for row in final_results:
            f.write(json.dumps(row) + "\n")

    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
