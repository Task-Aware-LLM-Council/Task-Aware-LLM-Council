"""P4 benchmark runner — joint decomposer+router + quantized specialists + synth.

Runs the task-aware LLM council's P4 policy (`LearnedRouterPolicy`) end-to-end
on CARC with:
  - Joint decomposer+router (`artifacts/decomposer_router_model/`) in-process
    via `HFSeq2SeqGenerate` — loads first, stays resident (~150MB).
  - Quantized 3-role specialist stack via `build_default_local_vllm_orchestrator_config()`
    — `task-aware-llm-council/{gemma-2-9b-it-GPTQ, DeepSeek-R1-Distill-Qwen-7B-AWQ-2,
    Qwen2.5-14B-Instruct-AWQ-2}`. Opened/closed by `PolicyRuntime`.
  - Synthesizer on port 8004 (DeepSeek-R1-Distill-Qwen-7B-AWQ-2, compressed-tensors).

Lifecycle mirrors the peer's `test_orchestartor_client.py`:
  joint router (in-process) → specialists (vLLM) → synthesizer (vLLM), each vLLM
  phase opened on demand by `PolicyRuntime` and torn down at exit.

Usage
-----
  uv run --package council-policies python p4_benchmark_client.py \\
    --model-dir artifacts/decomposer_router_model \\
    --dataset task-aware-llm-council/router_dataset \\
    --split test \\
    --limit 25 \\
    --out p4_results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from common import get_current_user
from council_policies import (
    HFSeq2SeqGenerate,
    LearnedRouterPolicy,
    Seq2SeqDecomposerRouter,
)
from council_policies.policy_runner import (
    CouncilBenchmarkRunner,
    PolicyRuntime,
)
from datasets import load_dataset
from llm_gateway import PromptRequest, Provider, ProviderConfig
from model_orchestration import (
    ModelSpec,
    OrchestratorConfig,
    build_default_local_vllm_orchestrator_config,
)


SYNTHESIZER_ROLE = "synthesizer"
SYNTHESIZER_MODEL = "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"


def build_synthesizer_config() -> OrchestratorConfig:
    """Port-8004 DeepSeek-AWQ synthesizer. Lifted from the reference script
    on `main` (test_orchestartor_client.py) — keep ports/params aligned so
    two benchmarks can't collide on the same vLLM port if run back-to-back."""
    return OrchestratorConfig(
        models=(
            ModelSpec(
                role=SYNTHESIZER_ROLE,
                model=SYNTHESIZER_MODEL,
                aliases=(SYNTHESIZER_ROLE,),
                provider_config=ProviderConfig(
                    provider=Provider.LOCAL,
                    default_model=SYNTHESIZER_MODEL,
                    default_params={
                        "local_launch_image": "vllm-openai_latest.sif",
                        "local_launch_port": 8004,
                        "local_launch_bind": f"/scratch1/{get_current_user()}/.cache",
                        "local_launch_startup_timeout_seconds": 600.0,
                        "local_launch_gpu_memory_utilization": 0.50,
                        "local_launch_quantization": "compressed-tensors",
                        "local_launch_use_gpu": True,
                    },
                ),
            ),
        ),
        default_role=SYNTHESIZER_ROLE,
        mode_label="local",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", required=True,
                   help="Path/HF ID of trained joint decomposer+router "
                        "(e.g. artifacts/decomposer_router_model).")
    p.add_argument("--dataset", default="task-aware-llm-council/router_dataset",
                   help="HF dataset name for evaluation prompts.")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=25,
                   help="Cap rows for smoke runs. None-equivalent via --limit -1.")
    p.add_argument("--out", type=Path, default=Path("p4_results.jsonl"))
    p.add_argument("--max-subtasks", type=int, default=4)
    return p.parse_args()


def build_requests(dataset_name: str, split: str, limit: int) -> list[dict]:
    ds = load_dataset(dataset_name, split=split)
    rows: list[dict] = []
    for index, row in enumerate(ds):
        if limit >= 0 and index >= limit:
            break
        rows.append({
            "index": index,
            "question": row["question"],
            "context": row.get("context", "") or "",
            "gold_answer": row.get("gold_answer"),
            "gold_label": row.get("gold_label"),
        })
    return rows


async def main() -> int:
    args = parse_args()

    # Joint decomposer+router: in-process, loads first, stays resident through
    # the full run. ~150MB on fp16 CUDA; tolerated next to the specialist stack.
    print(f"Loading joint decomposer+router from {args.model_dir}")
    generate_fn = HFSeq2SeqGenerate(args.model_dir)
    decomposer = Seq2SeqDecomposerRouter(
        generate_fn=generate_fn,
        max_subtasks=args.max_subtasks,
    )

    policy = LearnedRouterPolicy(
        decomposer=decomposer,
        synthesizer_role=SYNTHESIZER_ROLE,
        use_joint_model=True,
        max_subtasks=args.max_subtasks,
    )

    specialist_config = build_default_local_vllm_orchestrator_config()
    synthesizer_config = build_synthesizer_config()

    rows = build_requests(args.dataset, args.split, args.limit)
    print(f"Built {len(rows)} prompt requests from {args.dataset}:{args.split}")

    requests = [
        PromptRequest(
            user_prompt=row["question"],
            context=row["context"],
        )
        for row in rows
    ]

    async with PolicyRuntime(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
    ) as runtime:
        runner = CouncilBenchmarkRunner(policies=(policy,), runtime=runtime)
        print("Running P4 policy — specialists open lazily, synthesizer on demand")
        benchmark_result = await runner.run(requests)

    with args.out.open("w", encoding="utf-8") as f:
        for row, result in zip(rows, benchmark_result.results, strict=True):
            f.write(json.dumps({
                **row,
                "p4_answer": result.response.text,
                "predicted_route": result.metadata.get("predicted_route"),
                "synthesis_used": result.metadata.get("synthesis_used"),
                "error": result.metrics.error,
                "latency_ms": result.metrics.latency_ms,
            }) + "\n")

    print(f"Wrote {len(benchmark_result.results)} results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
