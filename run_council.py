#!/usr/bin/env python3
"""
Council runner for 3 vLLM models on CARC.

Local vLLM usage (called from run_council_job.sh):
  uv run python run_council.py \
    --ports 8000 8001 8002 \
    --question "What is the capital of France?"
"""
from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from council_policies import ModelConfig, run_council
from llm_gateway.models import Provider, ProviderConfig

load_dotenv()

COUNCIL_MODELS = [
    "task-aware-llm-council/gemma-2-9b-it-GPTQ",
    "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2",
    "task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2",
]


async def main(question: str, ports: list[int]) -> None:
    model_configs = [
        ModelConfig(
            model_id=model,
            provider_config=ProviderConfig(
                provider=Provider.LOCAL,
                api_base=f"http://127.0.0.1:{port}/v1/chat/completions",
            ),
        )
        for model, port in zip(COUNCIL_MODELS, ports)
    ]

    print(f"\nQuestion: {question}\n")
    print("Running council...\n")

    result = await run_council(model_configs, question)

    print("=== Answers ===")
    for label, ans in zip(["A", "B", "C"], result.answers):
        print(f"[{label}] {ans.model_config.model_id}:\n  {ans.answer[:300]}\n")

    print("=== Votes ===")
    for vote in result.votes:
        print(f"  {vote.voter_model_id} → {vote.voted_for} ({vote.confidence}): {vote.reason[:100]}")

    print(f"\n=== Winner: [{result.winner_label}] {result.winning_model_id} ===")
    if result.tiebreak_used:
        print("(Tiebreak was used)")
    print(f"\n{result.winning_answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Council")
    parser.add_argument("--question", required=True, help="Question to ask the council")
    parser.add_argument("--ports", nargs=3, type=int, default=[8000, 8001, 8002],
                        metavar="PORT", help="Ports for the 3 vLLM servers (default: 8000 8001 8002)")
    args = parser.parse_args()

    asyncio.run(main(args.question, args.ports))
