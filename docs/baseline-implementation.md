# P1 & P2 Step-by-Step Implementation Guide

**Scope:** This document is a hands-on engineering playbook for implementing Policy P1 (Single Best Model Baseline) and Policy P2 (Flat Council). It covers environment setup, pipeline architecture, code structure, execution steps, and debugging — everything needed to go from "repo cloned" to "results in hand."

**Prerequisites completed:** Dataset curation (5 individual + mixture), CARC environment with `uv` + Python 3.14, repo cloned on `remote-om` branch.

---

## STEP 0: Project Environment & Configuration

### 0.1 Directory Setup

Run this once on CARC to scaffold the project:

```bash
cd /home1/oghag/Task-Aware-LLM-Council

# Create the full directory tree
mkdir -p src/{api,data,prompts/{task_prompts,role_prompts},evaluation,tools,policies,analysis,utils}
mkdir -p scripts
mkdir -p results/{p1_single_best/{raw_responses,scores,figures},p2_flat_council/{raw_responses,scores,figures,voting_logs}}
mkdir -p models
mkdir -p notebooks

# Create __init__.py files so Python treats these as packages
find src -type d -exec touch {}/__init__.py \;

# Create .env file for secrets (add to .gitignore!)
echo "OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE" > .env
echo ".env" >> .gitignore
echo "results/" >> .gitignore
```

### 0.2 Dependencies

Add these to `pyproject.toml` under `[project.dependencies]`:

```toml
[project]
name = "task-aware-llm-council"
requires-python = ">=3.14"
dependencies = [
    # API & async
    "httpx>=0.27",
    "aiohttp>=3.9",
    "python-dotenv>=1.0",

    # Datasets
    "datasets>=2.19",
    "huggingface-hub>=0.23",

    # Evaluation metrics
    "rouge-score>=0.1.2",
    "bert-score>=0.3.13",
    "nltk>=3.8",

    # Analysis & plotting
    "pandas>=2.2",
    "matplotlib>=3.9",
    "seaborn>=0.13",
    "numpy>=1.26",

    # Router training (for P4 later, but install now)
    "sentence-transformers>=3.0",
    "scikit-learn>=1.5",

    # Utilities
    "tqdm>=4.66",
    "tenacity>=8.3",      # retry logic
    "tiktoken>=0.7",      # token counting
]
```

Then sync:

```bash
source .venv/bin/activate
uv sync
```

### 0.3 Configuration File (`src/utils/config.py`)

This is the single source of truth for models, datasets, and settings:

```python
"""
Central configuration for all experiments.
Edit MODEL_POOL when the final 16 models are confirmed.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API ──────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_CONCURRENT_REQUESTS = 10   # semaphore limit
REQUEST_TIMEOUT = 120          # seconds
MAX_RETRIES = 3

# ── PATHS ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"

# ── MODEL POOL (16 models) ───────────────────────────
#
# ⚠️  IMPORTANT — READ BEFORE RUNNING:
#
# Models #2 and #3 (Llama Prompt Guard) are CLASSIFIER models, not
# generative LLMs. They output binary labels (safe/unsafe), NOT text
# answers. They CANNOT answer QA, math, code, or fact-verification
# prompts. You must either:
#   (a) Remove them from P1/P2 benchmarking and use them only as a
#       safety filter in the council pipeline (P3/P4), OR
#   (b) Replace them with two generative models for benchmarking.
#
# Model #8 (THUDM) — "Thudm" is the org name (Tsinghua University),
# not a model name. The likely intended model is GLM-4 or a
# GLM-Z1-series math model. Using glm-z1-32b (reasoning/math) below.
# Adjust if your team intended a different THUDM model.
#
# Models #5, #6 (Qwen2.5 math/coder) — mapped to the 32B instruct
# variants on OpenRouter.
#
# Model #16 (Gemini) — mapped to google/gemini-3-pro-preview as the
# latest reasoning-capable Gemini on OpenRouter. Adjust if needed.
#

MODEL_POOL = [
    "qwen/qwen3-32b",                          #  1. Qwen3-32B (coding + general reasoning)
    # "meta-llama/llama-prompt-guard-2-22m",    #  2. ⛔ CLASSIFIER — skip for P1/P2
    # "meta-llama/llama-prompt-guard-2-86m",    #  3. ⛔ CLASSIFIER — skip for P1/P2
    "openai/gpt-oss-120b",                      #  4. GPT-OSS-120B (reasoning, writing, coding)
    "qwen/qwen-2.5-72b-instruct",              #  5. Qwen2.5 72B (math)
    "qwen/qwen-2.5-coder-32b-instruct",        #  6. Qwen2.5 Coder 32B (code)
    "deepseek/deepseek-r1",                     #  7. DeepSeek-R1 (math + reasoning)
    "z-ai/glm-z1-32b",                          #  8. THUDM / GLM-Z1-32B (math) — verify on OpenRouter
    "qwen/qwq-32b",                             #  9. QwQ-32B (math reasoning)
    "anthropic/claude-opus-4",                   # 10. Claude Opus (code generation) — verify exact ID
    "minimax/minimax-m1",                        # 11. MiniMax-M1 (critical thinking)
    "qwen/qwen3-30b-a3b",                       # 12. Qwen3-30B-A3B (RAG)
    "deepseek/deepseek-v3.2",                   # 13. DeepSeek V3.2 (coding) — verify exact ID
    "z-ai/glm-5",                                # 14. GLM-5 / Zhipu AI (coding)
    "moonshotai/kimi-k2.5",                      # 15. Kimi K2.5 / Moonshot (coding)
    "google/gemini-3-pro-preview",               # 16. Gemini 3 Pro (reasoning)
]

# With Prompt Guard models excluded, this is 14 generative models.
# If you want exactly 16, consider adding two from:
#   "deepseek/deepseek-v3.1"         — strong general/coding
#   "mistralai/mistral-large"        — good all-rounder
#   "meta-llama/llama-3.3-70b-instruct" — strong open-source
#   "minimax/minimax-m2.5"           — strong coding/agents

# Short names for file paths and display
MODEL_SHORT_NAMES = {
    "qwen/qwen3-32b":                      "qwen3_32b",
    "openai/gpt-oss-120b":                 "gpt_oss_120b",
    "qwen/qwen-2.5-72b-instruct":         "qwen25_72b",
    "qwen/qwen-2.5-coder-32b-instruct":   "qwen25_coder",
    "deepseek/deepseek-r1":                "deepseek_r1",
    "z-ai/glm-z1-32b":                     "glm_z1_32b",
    "qwen/qwq-32b":                        "qwq_32b",
    "anthropic/claude-opus-4":              "claude_opus",
    "minimax/minimax-m1":                   "minimax_m1",
    "qwen/qwen3-30b-a3b":                  "qwen3_30b",
    "deepseek/deepseek-v3.2":              "deepseek_v32",
    "z-ai/glm-5":                           "glm5",
    "moonshotai/kimi-k2.5":                "kimi_k25",
    "google/gemini-3-pro-preview":          "gemini3_pro",
}

# ── DATASETS ─────────────────────────────────────────
DATASETS = {
    "musique": {
        "skill_tag": "multi_hop_qa",
        "metrics": ["exact_match", "token_f1"],
        "primary_metric": "token_f1",
    },
    "quality": {
        "skill_tag": "long_context_qa",
        "metrics": ["exact_match", "token_f1"],
        "primary_metric": "token_f1",
    },
    "fever": {
        "skill_tag": "fact_verification",
        "metrics": ["label_accuracy"],
        "primary_metric": "label_accuracy",
    },
    "hardmath": {
        "skill_tag": "math",
        "metrics": ["numeric_accuracy"],
        "primary_metric": "numeric_accuracy",
    },
    "humaneval_plus": {
        "skill_tag": "code",
        "metrics": ["pass_at_1"],
        "primary_metric": "pass_at_1",
    },
}

DATASET_NAMES = list(DATASETS.keys())

# ── EXPERIMENT SETTINGS ──────────────────────────────
PILOT_SAMPLES_PER_DATASET = 50       # for quick test runs
FULL_SAMPLES_PER_DATASET = 160       # ~800 total
VALIDATION_SPLIT_RATIO = 0.2         # 20% for P1 model selection

# ── P2: FLAT COUNCIL ─────────────────────────────────
COUNCIL_SIZE = 5                      # number of models in the council
COUNCIL_TEMPERATURES = [0.3, 0.5, 0.7, 0.3, 0.5]  # one per council member
```

### 0.4 Verify Network Access from CARC

**This is critical — do this before writing any code:**

```bash
# On CARC login node
curl -s -o /dev/null -w "%{http_code}" https://openrouter.ai/api/v1/models

# If you get 200 → outbound HTTPS works
# If you get 000 or timeout → API calls must be run locally
```

If CARC blocks outbound HTTPS, you'll need to run API scripts locally (on your laptop via WSL2) and `scp` results to CARC for analysis. Plan for this now.

---

## STEP 1: OpenRouter API Client

This is the foundation everything else depends on. Build it first, test it thoroughly.

### 1.1 Core Client (`src/api/openrouter_client.py`)

```python
"""
Async OpenRouter API client with:
- Concurrent requests with semaphore
- Automatic retries with exponential backoff
- Response caching to disk
- Token/cost tracking per request
"""
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    CACHE_DIR,
)

class OpenRouterClient:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.cost_log: list[dict] = []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cache helpers ────────────────────────────────
    def _cache_key(self, model: str, messages: list, **kwargs) -> str:
        """Deterministic cache key from request params."""
        payload = json.dumps({"model": model, "messages": messages, **kwargs}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return CACHE_DIR / f"{cache_key}.json"

    def _load_cache(self, cache_key: str) -> Optional[dict]:
        path = self._cache_path(cache_key)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _save_cache(self, cache_key: str, response: dict):
        self._cache_path(cache_key).write_text(json.dumps(response, indent=2))

    # ── Single model query ───────────────────────────
    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(min=2, max=30))
    async def query(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Query a single model. Returns cached response if available."""

        # Prepend system prompt if provided
        full_messages = messages.copy()
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + full_messages

        # Check cache
        cache_key = self._cache_key(model, full_messages, temperature=temperature)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        # Make the API call
        async with self.semaphore:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                start_time = time.time()
                resp = await client.post(
                    OPENROUTER_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": full_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                elapsed = time.time() - start_time

        resp.raise_for_status()
        data = resp.json()

        # Extract response content and usage
        result = {
            "model": model,
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
            "latency_seconds": elapsed,
            "timestamp": time.time(),
        }

        # Log cost
        usage = result["usage"]
        self.cost_log.append({
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency": elapsed,
        })

        # Cache to disk
        self._save_cache(cache_key, result)
        return result

    # ── Parallel query across multiple models ────────
    async def query_parallel(
        self,
        models: list[str],
        messages: list[dict],
        temperatures: Optional[list[float]] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Query multiple models concurrently."""
        if temperatures is None:
            temperatures = [0.0] * len(models)

        tasks = [
            self.query(model, messages, temperature=temp, system_prompt=system_prompt)
            for model, temp in zip(models, temperatures)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successes and failures
        successes = []
        for model, result in zip(models, results):
            if isinstance(result, Exception):
                print(f"  [FAIL] {model}: {result}")
            else:
                successes.append(result)

        return successes

    # ── Cost summary ─────────────────────────────────
    def get_cost_summary(self) -> dict:
        total_tokens = sum(e["total_tokens"] for e in self.cost_log)
        return {
            "total_requests": len(self.cost_log),
            "total_tokens": total_tokens,
            "total_prompt_tokens": sum(e["prompt_tokens"] for e in self.cost_log),
            "total_completion_tokens": sum(e["completion_tokens"] for e in self.cost_log),
            "avg_latency": sum(e["latency"] for e in self.cost_log) / max(len(self.cost_log), 1),
        }
```

### 1.2 Test the Client

**Do this immediately.** Don't write anything else until this works:

```python
# scripts/test_api.py
"""
Quick smoke test: send one query to one model, verify response.
Run: python scripts/test_api.py
"""
import asyncio
from src.api.openrouter_client import OpenRouterClient

async def main():
    client = OpenRouterClient()

    # Pick one model from your list (use the cheapest one for testing)
    test_model = "openai/gpt-4o-mini"  # or whatever is cheapest

    result = await client.query(
        model=test_model,
        messages=[{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
        temperature=0.0,
    )

    print(f"Model: {result['model']}")
    print(f"Response: {result['content']}")
    print(f"Tokens: {result['usage']}")
    print(f"Latency: {result['latency_seconds']:.2f}s")

    # Test caching: same call should be instant
    result2 = await client.query(
        model=test_model,
        messages=[{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
        temperature=0.0,
    )
    print(f"\nCached call latency: should be ~0s")

    # Test parallel
    results = await client.query_parallel(
        models=[test_model, test_model],
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        temperatures=[0.0, 0.7],
    )
    print(f"\nParallel results: {len(results)} responses")

    print(f"\nCost summary: {client.get_cost_summary()}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
cd /home1/oghag/Task-Aware-LLM-Council
source .venv/bin/activate
python scripts/test_api.py
```

**Checkpoint: Do not proceed until this test passes.** If it fails on CARC, test from your local WSL2 machine to isolate network vs. code issues.

---

## STEP 2: Dataset Loaders

### 2.1 Data Model (`src/data/sample.py`)

```python
"""Standardized sample format used across all datasets and policies."""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class BenchmarkSample:
    id: str                     # unique identifier
    dataset: str                # "musique" | "quality" | "fever" | "hardmath" | "humaneval_plus"
    skill_tag: str              # "multi_hop_qa" | "long_context_qa" | "fact_verification" | "math" | "code"
    prompt: str                 # formatted input to send to the model
    reference: str              # gold answer (string form)
    metadata: dict = field(default_factory=dict)  # dataset-specific extras
```

### 2.2 Loaders (`src/data/loaders.py`)

```python
"""
Dataset loaders. Each returns List[BenchmarkSample].

IMPORTANT: The team has already curated these datasets.
If you're loading from local files (not HuggingFace), adjust the
load paths to point to your curated data directory.
"""
import json
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from src.data.sample import BenchmarkSample
from src.utils.config import FULL_SAMPLES_PER_DATASET

def load_musique(n: Optional[int] = None) -> list[BenchmarkSample]:
    """Load MuSiQue multi-hop QA dataset."""
    # Option A: From HuggingFace
    ds = load_dataset("danlou/musique", split="validation")
    # Option B: From local curated file
    # ds = json.loads(Path("data/curated/musique.json").read_text())

    samples = []
    for i, item in enumerate(ds):
        if n and i >= n:
            break
        # Build the prompt with supporting paragraphs
        context = "\n\n".join(
            f"Paragraph {j+1}: {p['title']}\n{p['paragraph_text']}"
            for j, p in enumerate(item.get("paragraphs", []))
        )
        prompt = (
            f"Answer the following question using the provided paragraphs. "
            f"Give a short, direct answer.\n\n"
            f"{context}\n\n"
            f"Question: {item['question']}\n"
            f"Answer:"
        )
        samples.append(BenchmarkSample(
            id=f"musique_{i}",
            dataset="musique",
            skill_tag="multi_hop_qa",
            prompt=prompt,
            reference=item["answer"],
            metadata={"question": item["question"]},
        ))
    return samples

def load_quality(n: Optional[int] = None) -> list[BenchmarkSample]:
    """Load QuALITY long-context QA dataset."""
    ds = load_dataset("nyu-mll/quality", split="validation")

    samples = []
    for i, item in enumerate(ds):
        if n and i >= n:
            break
        # QuALITY has article + question + 4 options
        options = item.get("options", [])
        options_text = "\n".join(f"  ({chr(65+j)}) {opt}" for j, opt in enumerate(options))
        prompt = (
            f"Read the following article and answer the question.\n\n"
            f"Article:\n{item['article']}\n\n"
            f"Question: {item['question']}\n"
            f"Options:\n{options_text}\n\n"
            f"Answer with just the letter (A, B, C, or D):"
        )
        # Gold answer is typically 0-indexed
        gold_idx = item.get("gold_label", 0)
        reference = chr(65 + gold_idx) if isinstance(gold_idx, int) else str(gold_idx)
        samples.append(BenchmarkSample(
            id=f"quality_{i}",
            dataset="quality",
            skill_tag="long_context_qa",
            prompt=prompt,
            reference=reference,
            metadata={"question": item["question"]},
        ))
    return samples

def load_fever(n: Optional[int] = None) -> list[BenchmarkSample]:
    """Load FEVER fact verification dataset."""
    ds = load_dataset("fever/fever", "v1.0", split="labelled_dev")

    samples = []
    for i, item in enumerate(ds):
        if n and i >= n:
            break
        prompt = (
            f"Determine whether the following claim is Supported, Refuted, "
            f"or if there is Not Enough Information (NEI).\n\n"
            f"Claim: {item['claim']}\n\n"
            f"Answer with exactly one of: Supported, Refuted, NEI"
        )
        # Map label IDs to strings
        label_map = {0: "Supported", 1: "Refuted", 2: "NEI"}
        reference = label_map.get(item.get("label", -1), str(item.get("label", "")))
        samples.append(BenchmarkSample(
            id=f"fever_{i}",
            dataset="fever",
            skill_tag="fact_verification",
            prompt=prompt,
            reference=reference,
            metadata={"claim": item["claim"]},
        ))
    return samples

def load_hardmath(n: Optional[int] = None) -> list[BenchmarkSample]:
    """Load HARDMath competition math dataset."""
    # Adjust based on where your curated data lives
    # HARDMath may not be on HuggingFace — load from local JSON
    # ds = json.loads(Path("data/curated/hardmath.json").read_text())

    # Placeholder — replace with actual loading logic
    ds = load_dataset("jhu-clsp/HARDMath", split="test")

    samples = []
    for i, item in enumerate(ds):
        if n and i >= n:
            break
        prompt = (
            f"Solve the following math problem. Show your reasoning, then give "
            f"your final numerical answer on the last line after 'ANSWER: '.\n\n"
            f"Problem: {item['problem']}\n\n"
            f"Solution:"
        )
        samples.append(BenchmarkSample(
            id=f"hardmath_{i}",
            dataset="hardmath",
            skill_tag="math",
            prompt=prompt,
            reference=str(item.get("answer", "")),
            metadata={"problem": item["problem"]},
        ))
    return samples

def load_humaneval_plus(n: Optional[int] = None) -> list[BenchmarkSample]:
    """Load HumanEval+ code generation dataset."""
    ds = load_dataset("evalplus/humanevalplus", split="test")

    samples = []
    for i, item in enumerate(ds):
        if n and i >= n:
            break
        prompt = (
            f"Complete the following Python function. Return ONLY the function code, "
            f"no explanations.\n\n"
            f"{item['prompt']}"
        )
        samples.append(BenchmarkSample(
            id=f"humaneval_plus_{i}",
            dataset="humaneval_plus",
            skill_tag="code",
            prompt=prompt,
            reference=item.get("canonical_solution", ""),
            metadata={
                "task_id": item.get("task_id", ""),
                "test": item.get("test", ""),
                "entry_point": item.get("entry_point", ""),
            },
        ))
    return samples

def load_mixture(n_per_dataset: Optional[int] = None) -> list[BenchmarkSample]:
    """
    Load the curated mixture of all 5 datasets, shuffled.
    For P3/P4 routing evaluation — skill_tag is ground truth but
    the router must predict it from the prompt alone.
    """
    all_samples = []
    loaders = [load_musique, load_quality, load_fever, load_hardmath, load_humaneval_plus]

    for loader in loaders:
        samples = loader(n=n_per_dataset)
        all_samples.extend(samples)

    random.shuffle(all_samples)
    return all_samples

# ── Convenience: load a single dataset by name ──────
LOADER_MAP = {
    "musique": load_musique,
    "quality": load_quality,
    "fever": load_fever,
    "hardmath": load_hardmath,
    "humaneval_plus": load_humaneval_plus,
    "mixture": load_mixture,
}

def load_dataset_by_name(name: str, n: Optional[int] = None) -> list[BenchmarkSample]:
    """Load any dataset by name string."""
    if name not in LOADER_MAP:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(LOADER_MAP.keys())}")
    return LOADER_MAP[name](n=n)
```

### 2.3 Test the Loaders

```python
# scripts/test_loaders.py
"""Quick test: load 5 samples from each dataset, print stats."""
from src.data.loaders import LOADER_MAP

for name, loader in LOADER_MAP.items():
    if name == "mixture":
        continue
    samples = loader(n=5)
    print(f"\n{'='*60}")
    print(f"Dataset: {name} | Loaded: {len(samples)} samples")
    print(f"Skill tag: {samples[0].skill_tag}")
    print(f"Prompt preview (first 200 chars):\n  {samples[0].prompt[:200]}...")
    print(f"Reference: {samples[0].reference}")
```

**Checkpoint: Verify all 5 loaders return valid BenchmarkSample objects with non-empty prompts and references. Fix any HuggingFace loading issues before proceeding.**

> **IMPORTANT:** If your team has already curated the data into local JSON/JSONL files, modify each loader to read from your local path (e.g., `data/curated/musique.jsonl`) instead of calling `load_dataset()`. The BenchmarkSample output format stays the same either way.
> 

---

## STEP 3: Evaluation Metrics

### 3.1 Metrics Module (`src/evaluation/metrics.py`)

```python
"""
Scoring functions for all 5 task types.
Each function takes (prediction: str, reference: str) and returns a float.
"""
import re
import string
from collections import Counter

def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def exact_match(prediction: str, reference: str) -> float:
    """Normalized exact match. Returns 1.0 or 0.0."""
    return float(_normalize_text(prediction) == _normalize_text(reference))

def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score (SQuAD-style)."""
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()

    if not ref_tokens:
        return float(not pred_tokens)  # both empty = 1.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * (precision * recall) / (precision + recall)

def label_accuracy(prediction: str, reference: str) -> float:
    """For FEVER: exact match on label (Supported/Refuted/NEI)."""
    pred = prediction.strip().lower()
    ref = reference.strip().lower()

    # Handle common model output variations
    label_map = {
        "supported": "supported", "support": "supported", "true": "supported",
        "refuted": "refuted", "refute": "refuted", "false": "refuted",
        "nei": "nei", "not enough information": "nei", "unknown": "nei",
    }
    pred_normalized = label_map.get(pred, pred)
    ref_normalized = label_map.get(ref, ref)
    return float(pred_normalized == ref_normalized)

def numeric_accuracy(prediction: str, reference: str, tolerance: float = 0.01) -> float:
    """
    For HARDMath: extract final number from prediction, compare to reference.
    Tolerance is relative: |pred - ref| / max(|ref|, 1e-9) <= tolerance.
    """
    def extract_number(text: str) -> float | None:
        # Look for "ANSWER: <number>" pattern first
        match = re.search(r'ANSWER:\s*([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)', text)
        if match:
            return float(match.group(1))
        # Fallback: find the last number in the text
        numbers = re.findall(r'[-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?', text)
        if numbers:
            return float(numbers[-1])
        return None

    pred_num = extract_number(prediction)
    ref_num = extract_number(reference)

    if pred_num is None or ref_num is None:
        return 0.0

    if ref_num == 0:
        return float(abs(pred_num) < tolerance)

    relative_error = abs(pred_num - ref_num) / max(abs(ref_num), 1e-9)
    return float(relative_error <= tolerance)

def pass_at_1(prediction: str, test_code: str, entry_point: str, timeout: int = 10) -> float:
    """
    For HumanEval+: execute predicted code against test cases.
    Returns 1.0 if all tests pass, 0.0 otherwise.

    WARNING: Executes untrusted code. Use sandboxing in production.
    """
    import subprocess
    import tempfile

    # Combine prediction + test code
    full_code = prediction + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            f.flush()
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return float(result.returncode == 0)
    except (subprocess.TimeoutExpired, Exception):
        return 0.0

# ── Dispatcher: pick the right metric for a dataset ──
def evaluate_sample(prediction: str, sample) -> dict[str, float]:
    """
    Evaluate a prediction against a BenchmarkSample.
    Returns dict of {metric_name: score}.
    """
    from src.utils.config import DATASETS

    dataset_config = DATASETS[sample.dataset]
    scores = {}

    for metric_name in dataset_config["metrics"]:
        if metric_name == "exact_match":
            scores["exact_match"] = exact_match(prediction, sample.reference)
        elif metric_name == "token_f1":
            scores["token_f1"] = token_f1(prediction, sample.reference)
        elif metric_name == "label_accuracy":
            scores["label_accuracy"] = label_accuracy(prediction, sample.reference)
        elif metric_name == "numeric_accuracy":
            scores["numeric_accuracy"] = numeric_accuracy(prediction, sample.reference)
        elif metric_name == "pass_at_1":
            scores["pass_at_1"] = pass_at_1(
                prediction,
                sample.metadata.get("test", ""),
                sample.metadata.get("entry_point", ""),
            )

    return scores
```

### 3.2 Test the Metrics

```python
# scripts/test_metrics.py
from src.evaluation.metrics import *

# Exact match
assert exact_match("The Eiffel Tower", "the eiffel tower") == 1.0
assert exact_match("Paris", "London") == 0.0

# Token F1
assert token_f1("the big red dog", "the big red dog") == 1.0
assert token_f1("big red dog", "the big red dog") > 0.8

# Label accuracy
assert label_accuracy("Supported", "supported") == 1.0
assert label_accuracy("True", "Supported") == 1.0  # alias mapping

# Numeric accuracy
assert numeric_accuracy("ANSWER: 42.0", "42") == 1.0
assert numeric_accuracy("The answer is 3.14159", "3.14159") == 1.0

print("All metric tests passed!")
```

**Checkpoint: All assertions pass.**

---

## STEP 4: P1 — Single Best Model Baseline

### 4.1 The Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                    P1 PIPELINE FLOW                                 │
│                                                                    │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │ Load Dataset │──▶│ For each of  │──▶│ Query model via      │    │
│  │ (N samples)  │   │ 16 models:   │   │ OpenRouter API       │    │
│  └─────────────┘   └──────────────┘   └──────────┬───────────┘    │
│                                                   │                │
│                                                   ▼                │
│                                       ┌──────────────────────┐    │
│                                       │ Extract answer from   │    │
│                                       │ model response        │    │
│                                       └──────────┬───────────┘    │
│                                                   │                │
│                                                   ▼                │
│                                       ┌──────────────────────┐    │
│                                       │ Score against gold    │    │
│                                       │ reference             │    │
│                                       └──────────┬───────────┘    │
│                                                   │                │
│                                                   ▼                │
│                          ┌────────────────────────────────────┐    │
│                          │ Aggregate: 16×5 results matrix     │    │
│                          │ → Select best overall model        │    │
│                          │ → Select specialist per dataset    │    │
│                          └────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

### 4.2 Answer Extraction (`src/evaluation/answer_extraction.py`)

Models don't return clean answers — they return paragraphs with reasoning. You need to extract the actual answer:

```python
"""
Extract the final answer from model responses for each task type.
This is one of the most error-prone parts — test thoroughly.
"""
import re

def extract_answer(response: str, dataset: str) -> str:
    """Route to the right extractor based on dataset."""
    extractors = {
        "musique": extract_qa_answer,
        "quality": extract_mcq_answer,
        "fever": extract_fever_label,
        "hardmath": extract_math_answer,
        "humaneval_plus": extract_code,
    }
    return extractors[dataset](response)

def extract_qa_answer(response: str) -> str:
    """Extract short answer from QA response. Take the first line/sentence."""
    # If the model followed instructions, answer is short
    lines = response.strip().split('\n')
    # Take first non-empty line
    for line in lines:
        line = line.strip()
        if line and not line.startswith(("Let me", "Based on", "According")):
            return line
    return response.strip().split('\n')[0]

def extract_mcq_answer(response: str) -> str:
    """Extract letter answer (A/B/C/D) from MCQ response."""
    response = response.strip()
    # Direct letter answer
    match = re.match(r'^([A-D])\b', response)
    if match:
        return match.group(1)
    # "The answer is (B)" pattern
    match = re.search(r'(?:answer is|answer:)\s*\(?([A-D])\)?', response, re.IGNORECASE)
    if match:
        return match.group(1)
    # Last resort: find any standalone letter
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)
    return response[:1]  # fallback

def extract_fever_label(response: str) -> str:
    """Extract Supported/Refuted/NEI from response."""
    response_lower = response.lower().strip()
    if "supported" in response_lower or "support" in response_lower:
        return "Supported"
    if "refuted" in response_lower or "refute" in response_lower:
        return "Refuted"
    if "nei" in response_lower or "not enough" in response_lower:
        return "NEI"
    return response.strip().split()[0] if response.strip() else ""

def extract_math_answer(response: str) -> str:
    """Extract final numerical answer from math response."""
    # Look for explicit "ANSWER: X" pattern
    match = re.search(r'ANSWER:\s*(.*)', response)
    if match:
        return match.group(1).strip()
    # Look for "the answer is X"
    match = re.search(r'(?:the answer is|final answer[:\s]*)\s*([-+]?[\d]*\.?[\d]+)', response, re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: last number in the response
    numbers = re.findall(r'[-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?', response)
    return numbers[-1] if numbers else ""

def extract_code(response: str) -> str:
    """Extract Python code from response (strip markdown fences)."""
    # Remove markdown code fences
    code = re.sub(r'^```(?:python)?\s*\n?', '', response.strip())
    code = re.sub(r'\n?```\s*$', '', code)
    return code.strip()
```

### 4.3 Baseline Runner (`src/policies/single_best.py`)

```python
"""
P1: Run every model on every dataset, score, and identify best overall + specialists.
"""
import asyncio
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.api.openrouter_client import OpenRouterClient
from src.data.loaders import load_dataset_by_name
from src.evaluation.metrics import evaluate_sample
from src.evaluation.answer_extraction import extract_answer
from src.utils.config import (
    MODEL_POOL, MODEL_SHORT_NAMES, DATASET_NAMES, DATASETS,
    RESULTS_DIR, PILOT_SAMPLES_PER_DATASET, FULL_SAMPLES_PER_DATASET,
)

async def run_model_on_dataset(
    client: OpenRouterClient,
    model: str,
    dataset_name: str,
    samples: list,
    output_dir: Path,
) -> dict:
    """Run a single model on a single dataset. Returns aggregated scores."""

    model_short = MODEL_SHORT_NAMES.get(model, model.split("/")[-1])
    raw_dir = output_dir / "raw_responses" / model_short
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_scores = []

    for sample in tqdm(samples, desc=f"  {model_short} × {dataset_name}", leave=False):
        # Query model
        result = await client.query(
            model=model,
            messages=[{"role": "user", "content": sample.prompt}],
            temperature=0.0,
        )

        # Extract answer
        raw_response = result["content"]
        extracted = extract_answer(raw_response, sample.dataset)

        # Score
        scores = evaluate_sample(extracted, sample)
        scores["sample_id"] = sample.id

        # Save raw response
        raw_path = raw_dir / f"{sample.id}.json"
        raw_path.write_text(json.dumps({
            "sample_id": sample.id,
            "model": model,
            "prompt": sample.prompt[:500] + "...",  # truncate for storage
            "raw_response": raw_response,
            "extracted_answer": extracted,
            "reference": sample.reference,
            "scores": scores,
        }, indent=2))

        all_scores.append(scores)

    # Aggregate
    primary_metric = DATASETS[dataset_name]["primary_metric"]
    avg_score = sum(s[primary_metric] for s in all_scores) / len(all_scores)

    return {
        "model": model,
        "dataset": dataset_name,
        "primary_metric": primary_metric,
        "avg_score": avg_score,
        "num_samples": len(all_scores),
        "all_scores": all_scores,
    }

async def run_p1_baseline(pilot: bool = False):
    """Full P1 pipeline."""
    client = OpenRouterClient()
    n_samples = PILOT_SAMPLES_PER_DATASET if pilot else FULL_SAMPLES_PER_DATASET
    output_dir = RESULTS_DIR / "p1_single_best"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_matrix = []  # list of {model, dataset, avg_score}

    for dataset_name in DATASET_NAMES:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (loading {n_samples} samples)")
        print(f"{'='*60}")

        samples = load_dataset_by_name(dataset_name, n=n_samples)

        for model in MODEL_POOL:
            print(f"\n  Running: {model}")
            result = await run_model_on_dataset(
                client, model, dataset_name, samples, output_dir
            )
            results_matrix.append({
                "model": model,
                "model_short": MODEL_SHORT_NAMES.get(model, model.split("/")[-1]),
                "dataset": dataset_name,
                "primary_metric": result["primary_metric"],
                "avg_score": result["avg_score"],
                "num_samples": result["num_samples"],
            })

            # Save incremental checkpoint
            pd.DataFrame(results_matrix).to_csv(
                output_dir / "scores" / "results_matrix.csv", index=False
            )
            print(f"    → {result['primary_metric']}: {result['avg_score']:.4f}")

    # ── Analysis ─────────────────────────────────────
    df = pd.DataFrame(results_matrix)

    # Best model per dataset (specialist map)
    specialist_map = {}
    for dataset_name in DATASET_NAMES:
        subset = df[df["dataset"] == dataset_name].sort_values("avg_score", ascending=False)
        best = subset.iloc[0]
        runner_up = subset.iloc[1] if len(subset) > 1 else best
        specialist_map[dataset_name] = {
            "best_model": best["model"],
            "best_score": best["avg_score"],
            "runner_up_model": runner_up["model"],
            "runner_up_score": runner_up["avg_score"],
        }

    # Best overall model (avg rank across datasets)
    df["rank"] = df.groupby("dataset")["avg_score"].rank(ascending=False)
    avg_ranks = df.groupby("model")["rank"].mean().sort_values()
    best_overall = avg_ranks.index[0]

    # Save outputs
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(scores_dir / "results_matrix.csv", index=False)

    json.dump(specialist_map, open(scores_dir / "specialist_map.json", "w"), indent=2)
    json.dump(
        {"best_overall_model": best_overall, "avg_rank": float(avg_ranks.iloc[0]),
         "all_avg_ranks": {k: float(v) for k, v in avg_ranks.items()}},
        open(scores_dir / "best_model_selection.json", "w"), indent=2
    )
    json.dump(client.get_cost_summary(), open(scores_dir / "cost_summary.json", "w"), indent=2)

    # Print summary
    print(f"\n\n{'='*60}")
    print("P1 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nBest overall model: {best_overall} (avg rank: {avg_ranks.iloc[0]:.2f})")
    print(f"\nSpecialists:")
    for ds, info in specialist_map.items():
        print(f"  {ds}: {info['best_model']} ({info['best_score']:.4f})")
    print(f"\nCost: {client.get_cost_summary()}")

    return df, specialist_map, best_overall
```

### 4.4 CLI Entry Point (`scripts/run_baseline.py`)

```python
"""
CLI to run P1 baseline.
Usage:
    python scripts/run_baseline.py --pilot            # 50 samples per dataset
    python scripts/run_baseline.py --full             # all samples
    python scripts/run_baseline.py --pilot --samples 20  # custom sample count
"""
import argparse
import asyncio

from src.policies.single_best import run_p1_baseline
from src.utils.config import PILOT_SAMPLES_PER_DATASET

def main():
    parser = argparse.ArgumentParser(description="Run P1 Single Best Model baseline")
    parser.add_argument("--pilot", action="store_true", help="Run with reduced samples")
    parser.add_argument("--full", action="store_true", help="Run with full samples")
    parser.add_argument("--samples", type=int, help="Override samples per dataset")
    args = parser.parse_args()

    if args.samples:
        import src.utils.config as cfg
        cfg.PILOT_SAMPLES_PER_DATASET = args.samples
        cfg.FULL_SAMPLES_PER_DATASET = args.samples

    asyncio.run(run_p1_baseline(pilot=args.pilot))

if __name__ == "__main__":
    main()
```

### 4.5 Execution Order for P1

```
1. python scripts/test_api.py          ← verify API works
2. python scripts/test_loaders.py      ← verify datasets load
3. python scripts/test_metrics.py      ← verify scoring works
4. python scripts/run_baseline.py --pilot --samples 5    ← 5 samples, 1 model (smoke test)
5. python scripts/run_baseline.py --pilot                ← 50 samples, all 16 models
6. Review results/p1_single_best/scores/results_matrix.csv
7. python scripts/run_baseline.py --full                 ← full run (~160 samples each)
```

**Checkpoint: After step 5, you should have a results_matrix.csv with 16×5=80 rows, a specialist_map.json, and a best_model_selection.json. Review these before committing to the full run.**

---

## STEP 5: P2 — Flat Council

### 5.1 The Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        P2 PIPELINE FLOW                                  │
│                                                                         │
│  ┌─────────────┐   ┌────────────────────┐   ┌───────────────────────┐  │
│  │ Load Dataset │──▶│ Select council     │──▶│ For each sample:      │  │
│  │ (N samples)  │   │ (top-5 from P1     │   │                       │  │
│  └─────────────┘   │  or diverse set)    │   │  ┌─────────────────┐  │  │
│                     └────────────────────┘   │  │ Stage 1:        │  │  │
│                                              │  │ Query all 5     │  │  │
│                                              │  │ models parallel │  │  │
│                                              │  └────────┬────────┘  │  │
│                                              │           │           │  │
│                                              │           ▼           │  │
│                                              │  ┌─────────────────┐  │  │
│                                              │  │ Stage 2:        │  │  │
│                                              │  │ Aggregate via   │  │  │
│                                              │  │ majority vote   │  │  │
│                                              │  └────────┬────────┘  │  │
│                                              │           │           │  │
│                                              │           ▼           │  │
│                                              │  ┌─────────────────┐  │  │
│                                              │  │ Score council   │  │  │
│                                              │  │ answer + each   │  │  │
│                                              │  │ individual      │  │  │
│                                              │  └─────────────────┘  │  │
│                                              └───────────────────────┘  │
│                                                                         │
│  Output: council score vs P1 best model, per-dataset breakdown          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Council Selection Strategy

After P1 completes, you have two options for picking the 5 council members:

```python
def select_council_members(results_matrix_df, strategy="top5"):
    """
    Pick 5 models for the flat council.

    Strategies:
    - "top5": Top 5 by average rank (strongest overall)
    - "diverse": Best model per dataset + best overall (max coverage)
    """
    df = results_matrix_df.copy()
    df["rank"] = df.groupby("dataset")["avg_score"].rank(ascending=False)
    avg_ranks = df.groupby("model")["rank"].mean().sort_values()

    if strategy == "top5":
        return list(avg_ranks.index[:5])

    elif strategy == "diverse":
        # One specialist per dataset + fill remaining from top overall
        specialists = set()
        for dataset in df["dataset"].unique():
            best = df[df["dataset"] == dataset].sort_values("avg_score", ascending=False).iloc[0]
            specialists.add(best["model"])

        # If we have 5+ specialists, take the 5 with best avg rank
        if len(specialists) >= 5:
            specialist_ranks = avg_ranks[avg_ranks.index.isin(specialists)]
            return list(specialist_ranks.index[:5])

        # Otherwise, fill from top overall
        council = list(specialists)
        for model in avg_ranks.index:
            if model not in council:
                council.append(model)
            if len(council) >= 5:
                break
        return council[:5]
```

### 5.3 Aggregation Functions (`src/policies/aggregation.py`)

```python
"""
Aggregation strategies for combining council member responses into
a single council answer.
"""
from collections import Counter

from src.evaluation.answer_extraction import extract_answer

def majority_vote(responses: list[dict], dataset: str) -> str:
    """
    Extract answer from each response, return the most common one.
    Works for: MuSiQue, QuALITY, FEVER, HARDMath.
    """
    extracted = [extract_answer(r["content"], dataset) for r in responses]
    counter = Counter(extracted)
    return counter.most_common(1)[0][0]

def code_best_of_n(responses: list[dict], sample) -> str:
    """
    For HumanEval+: pick the first response that passes tests.
    If none pass, fall back to the response from the highest-ranked model.
    """
    from src.evaluation.metrics import pass_at_1

    for resp in responses:
        code = extract_answer(resp["content"], "humaneval_plus")
        passes = pass_at_1(
            code,
            sample.metadata.get("test", ""),
            sample.metadata.get("entry_point", ""),
        )
        if passes == 1.0:
            return code

    # Fallback: return first response
    return extract_answer(responses[0]["content"], "humaneval_plus")

def aggregate_council(responses: list[dict], sample) -> str:
    """Route to the right aggregation strategy based on dataset."""
    if sample.dataset == "humaneval_plus":
        return code_best_of_n(responses, sample)
    else:
        return majority_vote(responses, sample.dataset)
```

### 5.4 Flat Council Runner (`src/policies/flat_council.py`)

```python
"""
P2: Flat council — query N models on every sample, aggregate via voting.
"""
import asyncio
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.api.openrouter_client import OpenRouterClient
from src.data.loaders import load_dataset_by_name
from src.evaluation.metrics import evaluate_sample
from src.evaluation.answer_extraction import extract_answer
from src.policies.aggregation import aggregate_council
from src.utils.config import (
    DATASET_NAMES, DATASETS, RESULTS_DIR,
    COUNCIL_TEMPERATURES, PILOT_SAMPLES_PER_DATASET, FULL_SAMPLES_PER_DATASET,
)

async def run_p2_flat_council(
    council_models: list[str],
    pilot: bool = False,
):
    """Full P2 pipeline."""
    client = OpenRouterClient()
    n_samples = PILOT_SAMPLES_PER_DATASET if pilot else FULL_SAMPLES_PER_DATASET
    output_dir = RESULTS_DIR / "p2_flat_council"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save council config
    config = {
        "council_models": council_models,
        "temperatures": COUNCIL_TEMPERATURES[:len(council_models)],
        "council_size": len(council_models),
    }
    json.dump(config, open(output_dir / "council_config.json", "w"), indent=2)

    results = []
    individual_results = []

    for dataset_name in DATASET_NAMES:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        samples = load_dataset_by_name(dataset_name, n=n_samples)

        for sample in tqdm(samples, desc=f"  Council × {dataset_name}"):
            # ── Stage 1: Query all council members in parallel ──
            responses = await client.query_parallel(
                models=council_models,
                messages=[{"role": "user", "content": sample.prompt}],
                temperatures=COUNCIL_TEMPERATURES[:len(council_models)],
            )

            if not responses:
                print(f"    [WARN] No responses for {sample.id}, skipping")
                continue

            # ── Stage 2: Aggregate ──
            council_answer = aggregate_council(responses, sample)

            # ── Score council answer ──
            council_scores = evaluate_sample(council_answer, sample)

            # ── Also score each individual member (for comparison with P1) ──
            for resp in responses:
                individual_answer = extract_answer(resp["content"], sample.dataset)
                individual_scores = evaluate_sample(individual_answer, sample)
                individual_results.append({
                    "sample_id": sample.id,
                    "dataset": dataset_name,
                    "model": resp["model"],
                    **individual_scores,
                })

            # ── Log voting details ──
            voting_log = {
                "sample_id": sample.id,
                "dataset": dataset_name,
                "individual_answers": [
                    {"model": r["model"], "answer": extract_answer(r["content"], sample.dataset)}
                    for r in responses
                ],
                "council_answer": council_answer,
                "reference": sample.reference,
                "council_scores": council_scores,
            }

            # Save per-sample log
            log_dir = output_dir / "voting_logs" / dataset_name
            log_dir.mkdir(parents=True, exist_ok=True)
            json.dump(voting_log, open(log_dir / f"{sample.id}.json", "w"), indent=2)

            results.append({
                "sample_id": sample.id,
                "dataset": dataset_name,
                "council_answer": council_answer,
                "reference": sample.reference,
                **council_scores,
            })

        # Checkpoint after each dataset
        pd.DataFrame(results).to_csv(output_dir / "scores" / "council_scores.csv", index=False)

    # ── Save all results ─────────────────────────────
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    df_council = pd.DataFrame(results)
    df_individual = pd.DataFrame(individual_results)

    df_council.to_csv(scores_dir / "council_scores.csv", index=False)
    df_individual.to_csv(scores_dir / "individual_scores.csv", index=False)
    json.dump(client.get_cost_summary(), open(scores_dir / "cost_summary.json", "w"), indent=2)

    # ── Summary ──────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("P2 FLAT COUNCIL RESULTS")
    print(f"{'='*60}")

    for dataset_name in DATASET_NAMES:
        primary = DATASETS[dataset_name]["primary_metric"]
        subset = df_council[df_council["dataset"] == dataset_name]
        if primary in subset.columns:
            avg = subset[primary].mean()
            print(f"  {dataset_name} ({primary}): {avg:.4f}")

    print(f"\nCost: {client.get_cost_summary()}")
    return df_council, df_individual
```

### 5.5 CLI Entry Point (`scripts/run_flat_council.py`)

```python
"""
CLI to run P2 flat council.
Usage:
    python scripts/run_flat_council.py --pilot
    python scripts/run_flat_council.py --full
    python scripts/run_flat_council.py --pilot --strategy diverse
"""
import argparse
import asyncio
import json

import pandas as pd

from src.policies.flat_council import run_p2_flat_council
from src.policies.single_best import select_council_members
from src.utils.config import RESULTS_DIR

def main():
    parser = argparse.ArgumentParser(description="Run P2 Flat Council")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--strategy", choices=["top5", "diverse"], default="top5",
                        help="Council member selection strategy")
    parser.add_argument("--models", nargs="+", help="Override: manually specify council model IDs")
    parser.add_argument("--samples", type=int, help="Override samples per dataset")
    args = parser.parse_args()

    if args.samples:
        import src.utils.config as cfg
        cfg.PILOT_SAMPLES_PER_DATASET = args.samples
        cfg.FULL_SAMPLES_PER_DATASET = args.samples

    # Select council members
    if args.models:
        council_models = args.models
    else:
        # Load P1 results to select council
        p1_results = pd.read_csv(RESULTS_DIR / "p1_single_best" / "scores" / "results_matrix.csv")
        council_models = select_council_members(p1_results, strategy=args.strategy)

    print(f"Council members ({args.strategy}): {council_models}")
    asyncio.run(run_p2_flat_council(council_models, pilot=args.pilot))

if __name__ == "__main__":
    main()
```

### 5.6 Execution Order for P2

```
1. Ensure P1 is complete (results_matrix.csv exists)
2. python scripts/run_flat_council.py --pilot --samples 5     ← smoke test
3. python scripts/run_flat_council.py --pilot                 ← 50 samples
4. Review results/p2_flat_council/scores/council_scores.csv
5. Review voting_logs/ — are votes sensible? Is extraction working?
6. python scripts/run_flat_council.py --full                  ← full run
7. python scripts/run_flat_council.py --full --strategy diverse  ← compare strategies
```

---

## STEP 6: P1 vs P2 Comparison

### 6.1 Comparison Script (`scripts/compare_p1_p2.py`)

```python
"""
Compare P1 (single best) vs P2 (flat council) results.
Generates comparison table + figures.
"""
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import RESULTS_DIR, DATASETS, DATASET_NAMES

def compare():
    # Load results
    p1_matrix = pd.read_csv(RESULTS_DIR / "p1_single_best" / "scores" / "results_matrix.csv")
    p1_best = json.load(open(RESULTS_DIR / "p1_single_best" / "scores" / "best_model_selection.json"))
    p2_scores = pd.read_csv(RESULTS_DIR / "p2_flat_council" / "scores" / "council_scores.csv")

    best_model = p1_best["best_overall_model"]

    # Build comparison table
    comparison = []
    for dataset_name in DATASET_NAMES:
        primary = DATASETS[dataset_name]["primary_metric"]

        # P1 best model score on this dataset
        p1_row = p1_matrix[
            (p1_matrix["model"] == best_model) & (p1_matrix["dataset"] == dataset_name)
        ]
        p1_score = p1_row["avg_score"].values[0] if len(p1_row) > 0 else 0

        # P2 council score
        p2_subset = p2_scores[p2_scores["dataset"] == dataset_name]
        p2_score = p2_subset[primary].mean() if primary in p2_subset.columns else 0

        comparison.append({
            "dataset": dataset_name,
            "metric": primary,
            "P1_single_best": round(p1_score, 4),
            "P2_flat_council": round(p2_score, 4),
            "delta": round(p2_score - p1_score, 4),
            "council_wins": p2_score > p1_score,
        })

    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "p1_vs_p2_comparison.csv", index=False)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(DATASET_NAMES))
    width = 0.35

    p1_vals = [df[df["dataset"] == d]["P1_single_best"].values[0] for d in DATASET_NAMES]
    p2_vals = [df[df["dataset"] == d]["P2_flat_council"].values[0] for d in DATASET_NAMES]

    ax.bar([i - width/2 for i in x], p1_vals, width, label="P1: Single Best", color="#4C72B0")
    ax.bar([i + width/2 for i in x], p2_vals, width, label="P2: Flat Council", color="#DD8452")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Score (primary metric)")
    ax.set_title("P1 (Single Best Model) vs P2 (Flat Council)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_NAMES, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "p1_vs_p2_comparison.png", dpi=150)
    print(f"\nFigure saved to {RESULTS_DIR / 'p1_vs_p2_comparison.png'}")

if __name__ == "__main__":
    compare()
```

---

## STEP 7: Full Execution Checklist

The complete order of operations, from zero to P2 results:

```
PHASE 0: SETUP
  □ 0.1  Scaffold directories (mkdir -p ...)
  □ 0.2  Add dependencies to pyproject.toml → uv sync
  □ 0.3  Create src/utils/config.py with model pool placeholder
  □ 0.4  Set up .env with OPENROUTER_API_KEY
  □ 0.5  Verify CARC outbound HTTPS (curl test)

PHASE 1: API CLIENT
  □ 1.1  Implement src/api/openrouter_client.py
  □ 1.2  Run scripts/test_api.py → MUST PASS before continuing

PHASE 2: DATA
  □ 2.1  Implement src/data/sample.py (BenchmarkSample dataclass)
  □ 2.2  Implement src/data/loaders.py (all 5 + mixture)
  □ 2.3  Run scripts/test_loaders.py → verify all 5 datasets

PHASE 3: EVALUATION
  □ 3.1  Implement src/evaluation/metrics.py
  □ 3.2  Implement src/evaluation/answer_extraction.py
  □ 3.3  Run scripts/test_metrics.py → all assertions pass

PHASE 4: P1 BASELINE
  □ 4.1  Paste final 16 model IDs into config.py
  □ 4.2  Implement src/policies/single_best.py
  □ 4.3  Smoke test: run_baseline.py --pilot --samples 5
  □ 4.4  Pilot run: run_baseline.py --pilot (50 samples)
  □ 4.5  Review results_matrix.csv — sanity check scores
  □ 4.6  Full run: run_baseline.py --full
  □ 4.7  Verify specialist_map.json + best_model_selection.json

PHASE 5: P2 FLAT COUNCIL
  □ 5.1  Implement src/policies/aggregation.py
  □ 5.2  Implement src/policies/flat_council.py
  □ 5.3  Smoke test: run_flat_council.py --pilot --samples 5
  □ 5.4  Pilot run: run_flat_council.py --pilot
  □ 5.5  Review voting_logs/ — verify extraction + voting logic
  □ 5.6  Full run: run_flat_council.py --full
  □ 5.7  (Optional) Run with --strategy diverse for comparison

PHASE 6: COMPARISON
  □ 6.1  Run scripts/compare_p1_p2.py
  □ 6.2  Review p1_vs_p2_comparison.csv + figure
  □ 6.3  Commit results to repo
```

---

## Debugging Tips

**"Model returns empty/garbage"**
→ Check the raw response in `results/cache/`. Likely a prompt formatting issue. Test the exact prompt manually on OpenRouter's playground.

**"Scores are all 0.0"**
→ Answer extraction is probably failing. Check `answer_extraction.py` — print the raw response and extracted answer for a few samples. Most common issue: model wraps answer in explanation text.

**"API returns 429 (rate limited)"**
→ Reduce `MAX_CONCURRENT_REQUESTS` in config.py. Start with 5, increase gradually.

**"Different results on re-run"**
→ Cache is keyed on (model, messages, temperature). If you change prompts, old cache is stale. Clear `results/cache/` to force re-query.

**"CARC network timeout"**
→ Run API scripts on your local machine (WSL2), save results to `results/`, then `scp` to CARC for analysis.

---

*Once P1 and P2 are complete, the data for the specialist map feeds directly into P3 (rule-based routing) and P4 (learned routing). Those phases reuse the same API client, loaders, and metrics — only the orchestration policy changes.*