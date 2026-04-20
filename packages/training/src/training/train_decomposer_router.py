"""
Fine-tune Flan-T5-small into the joint decomposer + router.

Run:

    uv run -m training.train_decomposer_router --out artifacts/decomposer_router_model

Pipeline:
    1. Load `task-aware-llm-council/decomposer-router-dataset`
       (splits: train, dev, mini_test, gold_eval — built by
       `build_decomposer_router_dataset.py`).
    2. Featurize each input as
           "decompose_and_route: " + featurize(question, context)
       reusing `council_policies.router_featurize.featurize` so train-
       and serve-time inputs are byte-identical.
    3. Targets are compact JSON arrays:
           json.dumps(row["targets"], separators=(",",":"))
       matching the `_JSON_ARRAY_RE` that both the build pipeline and
       the serving class rely on.
    4. Fine-tune `google/flan-t5-small` with `Seq2SeqTrainer`. Early
       stop on `composite` (role-accuracy + JSON parseability + ROUGE-L).
    5. Eval on dev, mini_test, gold_eval. Write artifact + sibling
       `decomposer_router_card.json` + `generation_config.json`.

Heavy deps (`torch`, `transformers`, `datasets`, `numpy`) import
inside `main()`. Keeps the module grep-able without pulling the ML
stack into tooling.

Why seq2seq, not decoder-only
-----------------------------
Encoder-decoder models generalize structured-JSON generation from <1k
gold anchors far better than a comparably sized decoder-only model.
The tokenizer's `extra_id_*` tokens also give us a natural
prefix-constraint hook if role-string misspellings appear in eval (see
plan's "Risks" section).

Why not vLLM
------------
Flan-T5 / encoder-decoder models are not natively supported by vLLM
as of 2026 (RFC #7366 closed; T5's relative positional bias blocks
the existing attention backends). The serving class
`Seq2SeqDecomposerRouter` runs HuggingFace `transformers` in-process
via a `GenerateFn` adapter — no vLLM dependency on the serve path.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "task-aware-llm-council/decomposer-router-dataset"
DEFAULT_BASE_MODEL = "google/flan-t5-small"
DEFAULT_SEED = 42
INPUT_FORMAT_VERSION = "featurize_v1"

# Same JSON-array regex used in decomposer.py / build script — keep the
# three parsers in lockstep.
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass(slots=True)
class TrainArgs:
    out: Path
    dataset: str
    dataset_revision: str
    base_model: str
    epochs: int
    lr: float
    batch_size: int
    input_max_length: int
    target_max_length: int
    generation_num_beams: int
    generation_max_new_tokens: int
    warmup_ratio: float
    weight_decay: float
    seed: int


def parse_args(argv: list[str] | None = None) -> TrainArgs:
    p = argparse.ArgumentParser(
        description="Fine-tune Flan-T5-small into the joint decomposer+router."
    )
    p.add_argument("--out", type=Path, required=True, help="Artifact directory.")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument(
        "--dataset-revision",
        default="",
        help="HF dataset revision/SHA to pin. Empty = latest.",
    )
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--input-max-length", type=int, default=512)
    p.add_argument("--target-max-length", type=int, default=256)
    p.add_argument("--generation-num-beams", type=int, default=2)
    p.add_argument("--generation-max-new-tokens", type=int, default=256)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ns = p.parse_args(argv)
    return TrainArgs(
        out=ns.out,
        dataset=ns.dataset,
        dataset_revision=ns.dataset_revision,
        base_model=ns.base_model,
        epochs=ns.epochs,
        lr=ns.lr,
        batch_size=ns.batch_size,
        input_max_length=ns.input_max_length,
        target_max_length=ns.target_max_length,
        generation_num_beams=ns.generation_num_beams,
        generation_max_new_tokens=ns.generation_max_new_tokens,
        warmup_ratio=ns.warmup_ratio,
        weight_decay=ns.weight_decay,
        seed=ns.seed,
    )


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _model_version(base_model: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sha = _git_sha() or "nosha"
    base_short = base_model.split("/")[-1]
    return f"{today}-{base_short}-{sha}"


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Featurization
# --------------------------------------------------------------------------- #


def _featurize_row(
    row: dict[str, Any], featurize_fn: Any, *, input_prefix: str,
) -> str:
    """Serve-time-identical input format. `input_prefix` is passed in
    rather than imported at module top so this file stays importable
    without council_policies installed (e.g. for linting)."""
    base = featurize_fn(
        row.get("question") or "",
        row.get("context") or "",
    )
    return input_prefix + base


def _target_text(row: dict[str, Any]) -> str:
    """Compact JSON array (no spaces) matching `_JSON_ARRAY_RE`."""
    targets = row.get("targets") or []
    return json.dumps(targets, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _parse_generated(text: str) -> list[dict[str, str]] | None:
    """Parse a generated JSON array. None on any failure."""
    if not text:
        return None
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    cleaned: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        subtask = item.get("subtask")
        if not isinstance(role, str) or not isinstance(subtask, str):
            continue
        cleaned.append({"role": role.strip(), "subtask": subtask.strip()})
    return cleaned or None


def _rougeL(pred: str, target: str) -> float:
    """Cheap ROUGE-L implementation (no `rouge_score` dep).

    Token-level LCS / max(|pred|, |target|). Tokenization is whitespace
    + lowercasing — good enough for an eval metric on decomposed
    subtask strings. Return 0 on either side empty.
    """
    p_tokens = pred.lower().split()
    t_tokens = target.lower().split()
    if not p_tokens or not t_tokens:
        return 0.0
    n, m = len(p_tokens), len(t_tokens)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if p_tokens[i - 1] == t_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[m]
    return lcs / max(n, m)


def _compute_metrics_fn(tokenizer: Any, role_labels: tuple[str, ...]):
    """Return the `compute_metrics` callback for Seq2SeqTrainer.

    Metrics:
      - json_parseable: fraction of preds that parse to a list of
        {role, subtask} dicts
      - shape_match: fraction where len(pred) == len(target) (given parseable)
      - role_accuracy: per-slot role match, conditional on shape match
      - subtask_rougeL: avg ROUGE-L across aligned slots, conditional on
        shape match
      - composite: 0.5 * role_acc + 0.3 * json_parseable + 0.2 * rougeL
        — single scalar for `metric_for_best_model`.
    """
    import numpy as np

    role_set = set(role_labels)

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        preds, labels = eval_pred
        # `predict_with_generate=True` produces token IDs directly; the
        # pad token for labels is -100 → replace before decoding.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        n = len(decoded_preds)
        parseable = 0
        shape_matches = 0
        role_slots_total = 0
        role_slots_correct = 0
        rouge_slots_total = 0
        rouge_sum = 0.0

        for pred_text, label_text in zip(decoded_preds, decoded_labels):
            pred_parsed = _parse_generated(pred_text)
            label_parsed = _parse_generated(label_text)
            if pred_parsed is None:
                continue
            parseable += 1
            if label_parsed is None:
                # Gold shouldn't fail to parse, but don't crash if it does.
                continue
            if len(pred_parsed) != len(label_parsed):
                continue
            shape_matches += 1
            for p_item, l_item in zip(pred_parsed, label_parsed):
                role_slots_total += 1
                if (
                    p_item["role"] == l_item["role"]
                    and p_item["role"] in role_set
                ):
                    role_slots_correct += 1
                rouge_slots_total += 1
                rouge_sum += _rougeL(p_item["subtask"], l_item["subtask"])

        json_parseable = parseable / max(1, n)
        shape_match = shape_matches / max(1, n)
        role_accuracy = role_slots_correct / max(1, role_slots_total)
        subtask_rougeL = rouge_sum / max(1, rouge_slots_total)
        composite = (
            0.5 * role_accuracy
            + 0.3 * json_parseable
            + 0.2 * subtask_rougeL
        )
        return {
            "json_parseable": float(json_parseable),
            "shape_match": float(shape_match),
            "role_accuracy": float(role_accuracy),
            "subtask_rougeL": float(subtask_rougeL),
            "composite": float(composite),
        }

    return compute_metrics


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    _seed_everything(args.seed)

    import numpy as np
    import torch
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from council_policies import DecomposerRouterCard, INPUT_PREFIX
    from council_policies.router_featurize import (
        DEFAULT_CONTEXT_CHAR_CAP,
        featurize,
    )
    from council_policies.router_labels import ROLE_LABELS

    # --- Load splits ------------------------------------------------------ #
    load_kwargs: dict[str, Any] = {}
    if args.dataset_revision:
        load_kwargs["revision"] = args.dataset_revision
    logger.info("Loading dataset %s", args.dataset)
    parquet_dir = Path(args.dataset) / "parquet"
    if parquet_dir.is_dir():
        data_files = {
            name: str(parquet_dir / f"{name}.parquet")
            for name in ("train", "dev", "mini_test", "gold_eval")
        }
        raw = load_dataset("parquet", data_files=data_files)
    else:
        raw = load_dataset(args.dataset, **load_kwargs)
    for required in ("train", "dev", "mini_test", "gold_eval"):
        if required not in raw:
            raise KeyError(
                f"Dataset {args.dataset!r} missing split {required!r}; "
                f"got {list(raw.keys())}"
            )

    # --- Tokenize --------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def _prepare(split: Dataset) -> Dataset:
        def _map_fn(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            inputs: list[str] = []
            targets: list[str] = []
            for i in range(len(batch["question"])):
                row = {k: batch[k][i] for k in batch}
                inputs.append(
                    _featurize_row(row, featurize, input_prefix=INPUT_PREFIX)
                )
                targets.append(_target_text(row))
            model_inputs = tokenizer(
                inputs,
                max_length=args.input_max_length,
                truncation=True,
            )
            labels = tokenizer(
                text_target=targets,
                max_length=args.target_max_length,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return split.map(
            _map_fn,
            batched=True,
            remove_columns=split.column_names,
            desc="tokenize",
        )

    train_ds = _prepare(raw["train"])
    dev_ds = _prepare(raw["dev"])
    mini_ds = _prepare(raw["mini_test"])
    gold_eval_ds = _prepare(raw["gold_eval"])
    logger.info(
        "sizes — train=%d dev=%d mini_test=%d gold_eval=%d",
        len(train_ds), len(dev_ds), len(mini_ds), len(gold_eval_ds),
    )

    # --- Model + collator ------------------------------------------------- #
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out / "_trainer_state"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="composite",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.generation_max_new_tokens,
        generation_num_beams=args.generation_num_beams,
        logging_steps=20,
        report_to=[],
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_compute_metrics_fn(tokenizer, ROLE_LABELS),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # --- Final eval on all three held-out splits -------------------------- #
    dev_metrics = trainer.evaluate(dev_ds, metric_key_prefix="dev")
    mini_metrics = trainer.evaluate(mini_ds, metric_key_prefix="mini_test")
    gold_metrics = trainer.evaluate(gold_eval_ds, metric_key_prefix="gold_eval")

    # --- Save artifact + generation_config + card ------------------------- #
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    generation_config = {
        "num_beams": args.generation_num_beams,
        "max_new_tokens": args.generation_max_new_tokens,
        "do_sample": False,
    }
    (args.out / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2)
    )

    def _clean(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
        wanted = ("json_parseable", "shape_match", "role_accuracy",
                  "subtask_rougeL", "composite", "loss")
        out: dict[str, float] = {}
        for key in wanted:
            full = f"{prefix}_{key}"
            if full in metrics:
                out[key] = float(metrics[full])
        return out

    card = DecomposerRouterCard(
        model_version=_model_version(args.base_model),
        base_model=args.base_model,
        roles=ROLE_LABELS,
        dataset_revision=args.dataset_revision,
        input_format_version=INPUT_FORMAT_VERSION,
        generation_config=generation_config,
        metrics={
            "dev": _clean(dev_metrics, "dev"),
            "mini_test": _clean(mini_metrics, "mini_test"),
            "gold_eval": _clean(gold_metrics, "gold_eval"),
            "n_train": len(train_ds),
            "n_dev": len(dev_ds),
            "n_mini_test": len(mini_ds),
            "n_gold_eval": len(gold_eval_ds),
        },
        training_config={
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "input_max_length": args.input_max_length,
            "target_max_length": args.target_max_length,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "context_char_cap": DEFAULT_CONTEXT_CHAR_CAP,
            "seed": args.seed,
        },
        git_sha=_git_sha(),
    )
    card_path = args.out / "decomposer_router_card.json"
    card.save(card_path)

    logger.info("Saved artifact to %s", args.out)
    logger.info("Card: %s", card_path)
    print(json.dumps(card.metrics, indent=2))

    _ = (np, torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
