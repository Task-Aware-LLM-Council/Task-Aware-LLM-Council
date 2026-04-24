"""P4 benchmark runner — joint decomposer+router + quantized specialists + synth.

Runs the task-aware LLM council's P4 policy (`LearnedRouterPolicy`) end-to-end
on CARC with:
  - Joint decomposer+router in-process via `HFCausalGenerate` — loads first,
    stays resident. Default is zero-shot `google/gemma-2-2b-it` (~5GB bf16);
    swap `--model-dir` for a fine-tuned local artifact once trained.
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

import hashlib

from common import get_current_user
from council_policies import (
    HFCausalGenerate,
    LearnedRouterPolicy,
    Seq2SeqDecomposerRouter,
)
from council_policies.policy_runner import (
    CouncilBenchmarkRunner,
    PolicyRuntime,
)
from council_policies.router import Subtask
from datasets import load_dataset
from llm_gateway import PromptRequest, Provider, ProviderConfig
from model_orchestration import (
    ModelSpec,
    OrchestratorConfig,
    build_default_local_vllm_orchestrator_config,
)


SYNTHESIZER_ROLE = "synthesizer"
SYNTHESIZER_MODEL = "task-aware-llm-council/DeepSeek-R1-Distill-Qwen-7B-AWQ-2"

_DECOMPOSER_CACHE_PATH = Path("artifacts/p4_decomposer_cache.jsonl")


class _CachingDecomposer:
    """Disk-backed cache around Seq2SeqDecomposerRouter.decompose().
    Key: sha256(prompt || context). Survives process restarts — plan-phase
    crashes no longer re-spend decomposer time on already-seen prompts."""

    def __init__(
        self, inner: Seq2SeqDecomposerRouter, cache_path: Path,
    ) -> None:
        self._inner = inner
        self._cache_path = cache_path
        self._memory: dict[str, list[Subtask]] = {}
        self._load()

    def _load(self) -> None:
        if not self._cache_path.exists():
            return
        with self._cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._memory[entry["key"]] = [
                        Subtask(
                            text=st["text"],
                            order=st["order"],
                            suggested_role=st.get("suggested_role"),
                        )
                        for st in entry["subtasks"]
                    ]
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Decomposer cache: loaded {len(self._memory)} entries")

    @staticmethod
    def _key(prompt: str, context: str) -> str:
        return hashlib.sha256(
            f"{prompt}||{context}".encode("utf-8")
        ).hexdigest()

    def _append(self, key: str, subtasks: list[Subtask]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "key": key,
                "subtasks": [
                    {
                        "text": st.text,
                        "order": st.order,
                        "suggested_role": st.suggested_role,
                    }
                    for st in subtasks
                ],
            }) + "\n")

    async def decompose(
        self, prompt: str, context: str = "",
    ) -> list[Subtask]:
        key = self._key(prompt, context)
        if key in self._memory:
            return self._memory[key]
        subtasks = await self._inner.decompose(prompt, context)
        self._memory[key] = subtasks
        self._append(key, subtasks)
        return subtasks

    # Proxy any other attrs (e.g. max_subtasks) to the inner decomposer so
    # callers that poke at the seq2seq internals still work.
    def __getattr__(self, name: str):
        return getattr(self._inner, name)


# Mirrors MusiqueProfile in packages/task_eval (commit 3370c0c) — oracle
# context + strict CoT prompt. router_dataset flattened is_supporting away, so
# we rebuild the oracle context from the source dataset at startup.
_MUSIQUE_CONSTRAINED_PROMPT = (
    "You are a strict reading comprehension assistant. You must analyze the "
    "context and think step-by-step out loud before answering.\n\n"
    "RULES:\n"
    "1. You must ONLY use the information provided in the Context. Do NOT use "
    "general knowledge.\n"
    "2. The answer is ALWAYS hidden somewhere in the text. You must search "
    "carefully.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Write your step-by-step reasoning inside <scratchpad> tags. "
    "After you are done thinking, conclude your response on a new line with "
    "the exact format: 'Final Answer: <exact entity name>'. "
    "If the answer is completely missing, output "
    "'Final Answer: NOT PRESENT IN CONTEXT'."
)


# Mirrors FeverProfile.row_to_case — forces the specialist to commit to one of
# the three FEVER labels so extract_fever_label has a surface to match on.
_FEVER_CONSTRAINED_PROMPT = (
    "Context:\n{context}\n\n"
    "Claim: {question}\n\n"
    "Based on the provided context, verify the claim. "
    "Answer strictly with one of these three labels: "
    "SUPPORTS, REFUTES, or NOT ENOUGH INFO."
)


# Mirrors QualityProfile.row_to_case — keep the specialist from padding the
# answer with a full explanatory sentence (EM is 0 otherwise, F1 dragged down).
_QUALITY_CONSTRAINED_PROMPT = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer the question concisely with just the answer."
)


# Mirrors HardMathProfile.row_to_case — extract_math_answer only parses the
# last \boxed{...}, so without this instruction every math answer scores 0.
_HARDMATH_CONSTRAINED_PROMPT = (
    "{question}\n\n"
    "Please put your final answer enclosed in \\boxed{{}}."
)


# HumanEvalPlus — specialist must return the full function as real Python.
# Without this the QA-routed gemma specialist often writes prose or
# pseudo-code explanations, breaking `extract_code_answer` or yielding
# responses missing the required `def <entry_point>(...):` signature.
# The diagnostic counted 22/60 NO_ENTRY_POINT + 22/60 SYNTAX_ERROR rows
# with no template; wrapping like this should recover most.
_HUMANEVAL_CONSTRAINED_PROMPT = (
    "{question}\n\n"
    "Complete the function above. Return only valid Python enclosed in a "
    "```python ... ``` fenced code block. Do not add explanations, do not "
    "rename the function, and do not remove the original signature or "
    "docstring."
)


_MUSIQUE_ORACLE_CACHE_DIR = Path("artifacts")


def _musique_oracle_cache_path(source: str, split: str) -> Path:
    slug = f"{source}_{split}".replace("/", "__")
    return _MUSIQUE_ORACLE_CACHE_DIR / f"musique_oracle_map_{slug}.json"


def _build_musique_oracle_map(
    source: str = "bdsaglam/musique",
    split: str = "validation",
) -> dict[str, str]:
    """Build an {original_id -> oracle context} map from the upstream
    MuSiQue dataset. Oracle context = concatenation of paragraphs
    flagged `is_supporting=True`. Router_dataset strips this flag so we
    rebuild it here from the source.

    Falls back to first 4 paragraphs when no is_supporting flag exists
    (defensive against MuSiQue schema drift)."""
    cache_path = _musique_oracle_cache_path(source, split)
    if cache_path.exists():
        print(f"Using cached MuSiQue oracle map at {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print(f"Building MuSiQue oracle map from {source}:{split} ...")
    src = load_dataset(source, split=split)
    mapping: dict[str, str] = {}
    for row in src:
        paragraphs = row.get("paragraphs") or []
        supporting = [
            p.get("paragraph_text", p.get("text", ""))
            for p in paragraphs
            if isinstance(p, dict) and p.get("is_supporting") is True
        ]
        if not supporting:
            supporting = [
                p.get("paragraph_text", p.get("text", ""))
                for p in paragraphs[:4]
                if isinstance(p, dict)
            ]
        mapping[str(row.get("id", ""))] = "\n\n".join(
            p.strip() for p in supporting if p and p.strip()
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mapping), encoding="utf-8")
    print(f"Cached {len(mapping)} rows to {cache_path}")
    return mapping


def _load_done_indices(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    done: set[int] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


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
    p.add_argument("--model-dir", default="google/gemma-2-2b-it",
                   help="HF hub ID or local path of the joint decomposer+router "
                        "model. Defaults to google/gemma-2-2b-it for zero-shot "
                        "evaluation; swap for a fine-tuned artifact once trained.")
    p.add_argument("--peft-adapter", default=None,
                   help="Optional LoRA adapter directory (e.g. "
                        "artifacts/decomposer_router_causal/adapter) to load "
                        "on top of --model-dir. Omit for zero-shot.")
    p.add_argument("--dataset", default="task-aware-llm-council/router_dataset",
                   help="HF dataset name for evaluation prompts.")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=25,
                   help="Cap rows for smoke runs. None-equivalent via --limit -1.")
    p.add_argument("--out", type=Path, default=Path("p4_results.jsonl"))
    p.add_argument("--max-subtasks", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=60,
                   help="Flush results to disk after every N prompts. Each chunk "
                        "pays one spec+synth cold start (~8min) because the 44GB "
                        "A40 can't hold both simultaneously. Bigger chunk = fewer "
                        "cold starts, bigger blast radius on crash.")
    p.add_argument("--musique-oracle", dest="musique_oracle",
                   action="store_true", default=True,
                   help="Inject is_supporting-filtered oracle context from "
                        "bdsaglam/musique for MuSiQue rows (default). Without "
                        "oracle, the specialist sees router_dataset's full "
                        "distractor-packed context.")
    p.add_argument("--no-musique-oracle", dest="musique_oracle",
                   action="store_false",
                   help="Disable oracle injection (design B, distractor-full).")
    p.add_argument("--musique-source", default="bdsaglam/musique",
                   help="Upstream MuSiQue dataset to pull is_supporting flags from.")
    p.add_argument("--musique-split", default="validation")
    p.add_argument(
        "--single-subtask-sources", default="FEVER,HumanEvalPlus,HARDMATH",
        help="Comma-separated source_dataset names that should bypass the "
             "decomposer and be dispatched as a single subtask containing "
             "the full original prompt. FEVER needs this because decomposed "
             "sub-claims lose the claim-as-a-whole framing; HumanEvalPlus "
             "because the decomposer mangles function signatures; HARDMATH "
             "because splitting a math problem into sub-questions tends to "
             "discard the constraint that ties them together. Empty string "
             "to disable.",
    )
    p.add_argument(
        "--force-role-sources",
        default="HumanEvalPlus:math_code,HARDMATH:math_code,FEVER:fact_general",
        help="Comma-separated SOURCE:ROLE overrides that bypass the router "
             "entirely. HumanEvalPlus and HARDMATH default to math_code "
             "(DeepSeek-R1) because the joint router was observed sending "
             ">90% of code rows to qa_reasoning (gemma QA, not a math/code "
             "model); HARDMATH has the same mismatch — math problems need "
             "the reasoning specialist, not QA. FEVER defaults to "
             "fact_general (Qwen-14B) because the router sent 75% of rows "
             "to qa_reasoning (gemma-9B) when the larger generalist is "
             "the better fit for claim verification. Empty string to disable.",
    )
    return p.parse_args()


def build_requests(
    dataset_name: str, split: str, limit: int, skip_indices: set[int],
    musique_oracle: dict[str, str] | None = None,
    single_subtask_sources: frozenset[str] = frozenset(),
    force_role_by_source: dict[str, str] | None = None,
) -> list[dict]:
    force_role_by_source = force_role_by_source or {}
    ds = load_dataset(dataset_name, split=split)
    oracle_hits = 0
    oracle_misses = 0

    rows: list[dict] = []
    for index, row in enumerate(ds):
        if limit >= 0 and index >= limit:
            break
        if index in skip_indices:
            continue
        source = row.get("source_dataset", "")
        question = row["question"]
        context = row.get("context", "") or ""

        prompt_template: str | None = None
        if source == "MuSiQue":
            # Oracle context: lookup by original_id in the bdsaglam/musique
            # is_supporting-filtered map. Without oracle, router_dataset's
            # default context is distractor-full — the specialist must search
            # carefully via the CoT template.
            if musique_oracle is not None:
                original_id = row.get("original_id")
                oracle_ctx = musique_oracle.get(str(original_id)) if original_id else None
                if oracle_ctx:
                    context = oracle_ctx
                    oracle_hits += 1
                else:
                    oracle_misses += 1
            prompt_template = _MUSIQUE_CONSTRAINED_PROMPT
        elif source == "FEVER":
            prompt_template = _FEVER_CONSTRAINED_PROMPT
        elif source == "QuALITY":
            prompt_template = _QUALITY_CONSTRAINED_PROMPT
        elif source == "HARDMATH":
            prompt_template = _HARDMATH_CONSTRAINED_PROMPT
        elif source == "HumanEvalPlus":
            prompt_template = _HUMANEVAL_CONSTRAINED_PROMPT

        rows.append({
            "index": index,
            "source_dataset": source,
            "question": question,
            "context": context,
            "gold_answer": row.get("gold_answer"),
            "gold_label": row.get("gold_label"),
            "prompt_template": prompt_template,
            "force_single_subtask": source in single_subtask_sources,
            "force_role": force_role_by_source.get(source),
        })
    return rows


async def main() -> int:
    args = parse_args()

    # Joint decomposer+router: in-process, loads first, stays resident through
    # the full run. Pinned to CPU by default — Gemma-2-2B on GPU is ~5GB bf16,
    # which doesn't fit next to three quantized specialists on a 44GB A40.
    # CPU adds ~500-1500ms per decompose call, tolerated at smoke-bench scale.
    # Flip to device="cuda" once specialists stop co-residing (or int8 Gemma).
    print(f"Loading joint decomposer+router from {args.model_dir} "
          f"(adapter={args.peft_adapter or 'none'}, device=cpu)")
    generate_fn = HFCausalGenerate(
        args.model_dir, peft_adapter=args.peft_adapter, device="cpu",
    )
    decomposer = Seq2SeqDecomposerRouter(
        generate_fn=generate_fn,
        max_subtasks=args.max_subtasks,
    )
    decomposer = _CachingDecomposer(decomposer, _DECOMPOSER_CACHE_PATH)

    policy = LearnedRouterPolicy(
        decomposer=decomposer,
        synthesizer_role=SYNTHESIZER_ROLE,
        use_joint_model=True,
        max_subtasks=args.max_subtasks,
    )

    specialist_config = build_default_local_vllm_orchestrator_config()
    synthesizer_config = build_synthesizer_config()

    done_indices = _load_done_indices(args.out)
    if done_indices:
        print(f"Resuming: {len(done_indices)} rows already in {args.out}, skipping.")

    musique_oracle: dict[str, str] | None = None
    if args.musique_oracle:
        musique_oracle = _build_musique_oracle_map(
            args.musique_source, args.musique_split,
        )

    single_subtask_sources = frozenset(
        s.strip() for s in (args.single_subtask_sources or "").split(",") if s.strip()
    )
    if single_subtask_sources:
        print(
            f"Bypassing decomposer for sources: "
            f"{sorted(single_subtask_sources)}"
        )

    force_role_by_source: dict[str, str] = {}
    for entry in (args.force_role_sources or "").split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        src, role = entry.split(":", 1)
        force_role_by_source[src.strip()] = role.strip()
    if force_role_by_source:
        print(f"Router overrides: {force_role_by_source}")

    rows = build_requests(
        args.dataset, args.split, args.limit, done_indices,
        musique_oracle=musique_oracle,
        single_subtask_sources=single_subtask_sources,
        force_role_by_source=force_role_by_source,
    )
    print(f"Built {len(rows)} prompt requests from {args.dataset}:{args.split}")
    if musique_oracle is not None:
        musique_rows = sum(1 for r in rows if r.get("source_dataset") == "MuSiQue")
        print(
            f"MuSiQue oracle context injected "
            f"(map size: {len(musique_oracle)}, "
            f"MuSiQue rows in batch: {musique_rows})"
        )
    if not rows:
        print("Nothing to do — all rows already completed.")
        return 0

    def _request_metadata(row: dict) -> dict:
        meta: dict = {}
        if row["prompt_template"]:
            meta["specialist_prompt_template"] = row["prompt_template"]
        if row.get("force_single_subtask"):
            meta["force_single_subtask"] = True
        if row.get("force_role"):
            meta["force_role"] = row["force_role"]
        return meta

    requests = [
        PromptRequest(
            user_prompt=row["question"],
            context=row["context"],
            metadata=_request_metadata(row),
        )
        for row in rows
    ]

    chunk_size = max(1, args.chunk_size)
    total_written = 0

    async with PolicyRuntime(
        specialist_config=specialist_config,
        synthesizer_config=synthesizer_config,
    ) as runtime:
        runner = CouncilBenchmarkRunner(policies=(policy,), runtime=runtime)
        # 3 specs (~27GB) + synth (~22GB) can't co-reside on a 44GB A40, so
        # the runner tears specs down before launching synth each chunk.
        # Larger --chunk-size amortizes ~8min cold-start overhead over more
        # prompts.
        print(
            f"Running P4 policy over {len(rows)} prompts "
            f"in chunks of {chunk_size}"
        )

        for chunk_start in range(0, len(rows), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(rows))
            chunk_rows = rows[chunk_start:chunk_end]
            chunk_reqs = requests[chunk_start:chunk_end]
            print(f"  chunk {chunk_start}-{chunk_end} ({len(chunk_rows)} prompts)")

            benchmark_result = await runner.run(chunk_reqs)

            mode = "a" if args.out.exists() else "w"
            with args.out.open(mode, encoding="utf-8") as f:
                for row, result in zip(
                    chunk_rows, benchmark_result.results, strict=True,
                ):
                    record = {k: v for k, v in row.items() if k != "prompt_template"}
                    f.write(json.dumps({
                        **record,
                        "p4_answer": result.response.text,
                        "predicted_route": result.metadata.get("predicted_route"),
                        "synthesis_used": result.metadata.get("synthesis_used"),
                        "error": result.metrics.error,
                        "latency_ms": result.metrics.latency_ms,
                    }) + "\n")
            total_written += len(chunk_rows)
            print(f"  flushed — {total_written}/{len(rows)} total")

    print(f"Wrote {total_written} results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
