"""
Entry point for the P2 Dataset Council Policy.

Usage:
    uv run council-p2

Environment variables:
    OPENROUTER_API_KEY   — for OpenRouter provider
    OPENAI_API_KEY       — for OpenAI provider
    HUGGINGFACE_API_KEY  — for HuggingFace provider
    NVIDIA_API_KEY       — for NVIDIA NIM provider

Optional CLI args:
    --provider       openrouter | openai | huggingface | nvidia   (default: openrouter)
    --qa-model       model name for qa role
    --reasoning-model model name for reasoning role
    --general-model  model name for general role
    --n-per-dataset  questions per dataset                        (default: 5)
    --output         path to save results JSON                    (default: p2_results.json)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from model_orchestration import ModelOrchestrator, build_default_orchestrator_config
from llm_gateway import Provider

from council_policies.p2_policy import DatasetCouncilPolicy, P2PolicyResult

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"

_DEFAULT_MODELS = {
    Provider.OPENROUTER: {
        "qa": "meta-llama/llama-3.1-8b-instruct",
        "reasoning": "deepseek/deepseek-r1",
        "general": "qwen/qwen-2.5-72b-instruct",
    },
    Provider.OPENAI: {
        "qa": "gpt-4o-mini",
        "reasoning": "o1-mini",
        "general": "gpt-4o",
    },
    Provider.HUGGINGFACE: {
        "qa": "meta-llama/Llama-3.1-8B-Instruct",
        "reasoning": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "general": "Qwen/Qwen2.5-72B-Instruct",
    },
    Provider.OPENAI_COMPATIBLE: {
        "qa": "meta/llama-3.1-8b-instruct",
        "reasoning": "nvidia/llama-3.1-nemotron-70b-instruct",
        "general": "meta/llama-3.3-70b-instruct",
    },
}

_API_KEY_ENV = {
    Provider.OPENROUTER: "OPENROUTER_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.HUGGINGFACE: "HUGGINGFACE_API_KEY",
    Provider.OPENAI_COMPATIBLE: "NVIDIA_API_KEY",
}

_PROVIDER_MAP = {
    "openrouter": Provider.OPENROUTER,
    "openai": Provider.OPENAI,
    "huggingface": Provider.HUGGINGFACE,
    "nvidia": Provider.OPENAI_COMPATIBLE,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the P2 Dataset Council Policy")
    parser.add_argument("--provider", default="openrouter", choices=list(_PROVIDER_MAP))
    parser.add_argument("--api-base", default=None, help="Override API base URL (auto-set for nvidia)")
    parser.add_argument("--qa-model", default=None)
    parser.add_argument("--reasoning-model", default=None)
    parser.add_argument("--general-model", default=None)
    parser.add_argument("--n-per-dataset", type=int, default=5)
    parser.add_argument("--output", default="p2_results.json", help="Path to save results JSON")
    return parser.parse_args()


def _save_results(result: P2PolicyResult, args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "n_per_dataset": args.n_per_dataset,
        "total_succeeded": len(result.results),
        "total_skipped": len(result.skipped_question_ids),
        "skipped_question_ids": result.skipped_question_ids,
        "dataset_votes": [
            {
                "dataset_name": s.dataset_name,
                "winner": s.winner,
                "question_count": s.question_count,
                "scores_by_role": s.scores_by_role,
            }
            for s in result.dataset_votes
        ],
        "questions": [
            {
                "example_id": qr.case.example.example_id,
                "dataset_name": qr.case.example.dataset_name,
                "question": qr.case.example.question,
                "best_model": qr.best_answer.role if qr.best_answer else None,
                "best_answer": qr.best_answer.text if qr.best_answer else None,
                "answers": [
                    {"role": a.role, "text": a.text, "error": a.error}
                    for a in qr.answers
                ],
                "ratings": [
                    {
                        "rater_role": r.rater_role,
                        "error": r.error,
                        "scores": {
                            r.label_to_role[e.label]: e.score
                            for e in r.ratings
                            if e.label in r.label_to_role
                        },
                    }
                    for r in qr.ratings
                ],
            }
            for qr in result.results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {output_path.resolve()}")


async def _run(args: argparse.Namespace) -> None:
    provider = _PROVIDER_MAP[args.provider]
    defaults = _DEFAULT_MODELS[provider]
    api_base = args.api_base or (_NVIDIA_API_BASE if args.provider == "nvidia" else None)

    config = build_default_orchestrator_config(
        provider=provider,
        api_base=api_base,
        api_key_env=_API_KEY_ENV[provider],
        qa_model=args.qa_model or defaults["qa"],
        reasoning_model=args.reasoning_model or defaults["reasoning"],
        general_model=args.general_model or defaults["general"],
    )

    async with ModelOrchestrator(config) as orchestrator:
        policy = DatasetCouncilPolicy(
            orchestrator,
            n_per_dataset=args.n_per_dataset,
        )
        result = await policy.run()

    # ── Print dataset-level results ───────────────────────────────────────────
    print("\n=== Dataset Vote Summary ===")
    for summary in result.dataset_votes:
        scores = ", ".join(f"{r}={s:.1f}" for r, s in summary.scores_by_role.items())
        print(f"  {summary.dataset_name:<20} winner={summary.winner:<12} scores=({scores})")

    # ── Print per-question best answers ───────────────────────────────────────
    print("\n=== Per-Question Best Answers ===")
    for qr in result.results:
        best = qr.best_answer
        if best:
            preview = best.text[:80].replace("\n", " ")
            print(f"  [{qr.case.example.dataset_name}] {qr.case.example.example_id}")
            print(f"    best model : {best.role}")
            print(f"    answer     : {preview}...")
        else:
            print(f"  [{qr.case.example.dataset_name}] {qr.case.example.example_id} — no ratings")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nTotal: {len(result.results)} succeeded, {len(result.skipped_question_ids)} skipped.")

    # ── Save results to JSON ──────────────────────────────────────────────────
    _save_results(result, args)


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
