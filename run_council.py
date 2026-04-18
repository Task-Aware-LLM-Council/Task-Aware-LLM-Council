#!/usr/bin/env python3
"""
Council runner: pulls 10 questions from each dataset domain and runs the council.

Usage (from SLURM job):
  uv run python run_council.py --ports 8000 8001 8002
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from dotenv import load_dotenv

from council_policies import ModelConfig, run_council
from llm_gateway.models import Provider, ProviderConfig
from task_eval.registry import get_dataset_profile

load_dotenv()

COUNCIL_MODELS = [
    "task-aware-llm-council/gemma-2-9b-it-GPTQ",
    "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2",
    "task-aware-llm-council/Qwen2.5-14B-Instruct-AWQ-2",
]

DATASETS = [
    ("musique",       "validation"),
    ("quality",       "validation"),
    ("fever",         "validation"),
    ("hardmath",      "test"),
    ("humaneval_plus","test"),
]

QUESTIONS_PER_DATASET = 10


def load_questions(dataset_name: str, split: str, n: int) -> list[dict]:
    profile = get_dataset_profile(dataset_name, split=split)
    questions = []
    for case in islice(profile.iter_cases(), n):
        ex = case.example
        prompt = ex.question or ""
        if ex.context:
            prompt = f"{ex.context}\n\n{prompt}"
        questions.append({
            "example_id": ex.example_id,
            "dataset": dataset_name,
            "prompt": prompt,
        })
    return questions


async def run_one(question: dict, model_configs: list[ModelConfig]) -> dict:
    try:
        result = await run_council(model_configs, question["prompt"])
        return {
            "example_id": question["example_id"],
            "dataset": question["dataset"],
            "question": question["prompt"][:300],
            "winner_model": result.winning_model_id,
            "winner_label": result.winner_label,
            "winning_answer": result.winning_answer,
            "tiebreak_used": result.tiebreak_used,
            "answers": [
                {"label": lbl, "model": a.model_config.model_id, "answer": a.answer}
                for lbl, a in zip(["A", "B", "C"], result.answers)
            ],
            "votes": [
                {"voter": v.voter_model_id, "voted_for": v.voted_for,
                 "confidence": v.confidence, "reason": v.reason}
                for v in result.votes
            ],
            "error": None,
        }
    except Exception as e:
        return {
            "example_id": question["example_id"],
            "dataset": question["dataset"],
            "question": question["prompt"][:300],
            "error": str(e),
        }


async def main(ports: list[int]) -> None:
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

    all_questions = []
    for dataset_name, split in DATASETS:
        print(f"Loading {QUESTIONS_PER_DATASET} questions from {dataset_name}...")
        all_questions.extend(load_questions(dataset_name, split, QUESTIONS_PER_DATASET))

    print(f"\nRunning council on {len(all_questions)} questions...\n")

    results = []
    for i, question in enumerate(all_questions, 1):
        print(f"[{i}/{len(all_questions)}] {question['dataset']} — {question['prompt'][:80]}...")
        result = await run_one(question, model_configs)
        results.append(result)
        if result["error"]:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Winner: [{result['winner_label']}] {result['winner_model']}")

    output_path = Path("council_results") / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Council across datasets")
    parser.add_argument("--ports", nargs=3, type=int, default=[8000, 8001, 8002],
                        metavar="PORT", help="Ports for the 3 vLLM servers")
    args = parser.parse_args()

    asyncio.run(main(args.ports))
