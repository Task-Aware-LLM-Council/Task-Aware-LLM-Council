"""
P3 Policy evaluation CLI — uses dataset profiles for proper constrained_question
formatting, passes system_prompt ditto, and writes metric-ready JSONL output.

Each output record has:
  - example_id, dataset_name
  - question          : the constrained_question from the profile (as sent to model)
  - response_text     : model's raw response (key used by all profile scorers)
  - reference         : ground-truth dict as built by the profile
  - task_type         : P3 routing classification
  - routed_role       : which specialist handled it
  - failed / error    : routing/dispatch status
  - latency_ms
  - answerable        : (musique-only) whether question is answerable
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from llm_gateway import Provider
from model_orchestration import ModelOrchestrator, build_default_orchestrator_config

from council_policies.p3_policy import RuleBasedRoutingPolicy, load_all_profiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Specialist config (mirrors existing cli.py) ────────────────────────────────
api_specialist_config = build_default_orchestrator_config(
    provider=Provider.OPENAI_COMPATIBLE,
    api_base="https://integrate.api.nvidia.com/v1/chat/completions",
    qa_model="google/gemma-3-27b-it",
    reasoning_model="openai/gpt-oss-120b",
    general_model="qwen/qwen2.5-coder-32b-instruct",
    api_key_env="NVIDIA_API_KEY",
)

N_PER_DATASET = 5       # samples per profile
OUTPUT_FILE = "results_p3_eval.jsonl"


async def async_main() -> None:
    print("Loading dataset profiles...")
    profiles = load_all_profiles()

    async with ModelOrchestrator(api_specialist_config) as orchestrator:
        policy = RuleBasedRoutingPolicy(
            orchestrator,
            n_per_dataset=N_PER_DATASET,
        )

        print("Sampling cases from profiles (constrained_question applied by each profile)...")
        cases = policy.sample_cases(profiles)
        print(f"Sampled {len(cases)} cases total across {len(profiles)} profiles.")

        if not cases:
            print("No cases sampled — aborting.")
            return

        print("Running P3 routing policy...")
        result = await policy.run(cases)

    print("Building metric-ready output records...")
    output_records: list[dict] = []

    for qr in result.results:
        case = qr.case
        example = case.example

        # Pull per-case metadata that's useful for scoring (avoids dumping raw_row)
        answerable = case.metadata.get("answerable")  # musique only

        record: dict = {
            "example_id": example.example_id,
            "dataset_name": example.dataset_name,
            # constrained_question — exactly what the profile built and sent to the model
            "question": example.question,
            # system_prompt passed ditto from the profile (None unless a profile sets it)
            "system_prompt": example.system_prompt,
            # context passed ditto from the profile
            "context": example.context,
            # KEY field used by all profile scorers via _prediction_text()
            "response_text": qr.response.text if qr.response is not None else None,
            # ground-truth reference as built by the profile
            "reference": case.reference,
            # routing metadata
            "task_type": qr.task_type.value,
            "routed_role": qr.routed_role,
            "failed": qr.failed,
            "error": qr.error,
            "latency_ms": qr.latency_ms,
        }

        # Include answerable flag when present (needed by MusiqueProfile.score)
        if answerable is not None:
            record["answerable"] = answerable

        output_records.append(record)

    output_path = Path(OUTPUT_FILE)
    with open(output_path, "w") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(output_records)} records to {output_path.resolve()}")

    # ── Routing summary ────────────────────────────────────────────────────────
    print("\n=== Routing Summary ===")
    for summary in result.routing_summaries:
        print(f"\nDataset : {summary.dataset_name}")
        print(f"  Questions   : {summary.question_count}")
        print(f"  Roles       : {summary.role_counts}")
        print(f"  Task types  : {summary.task_type_counts}")
        if summary.avg_latency_ms is not None:
            print(f"  Avg latency : {summary.avg_latency_ms:.0f} ms")

    skipped = len(result.skipped_question_ids)
    succeeded = sum(1 for r in result.results if not r.failed)
    print(f"\nTotal: {succeeded} succeeded, {skipped} skipped/failed.")


def main() -> None:
    print("Starting P3 policy evaluation CLI...")
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
