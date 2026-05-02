"""Fine-tune a causal chat LM (Gemma-2-2B-it) into the joint decomposer+router
via LoRA. Sibling of `train_decomposer_router.py`, which trains Flan-T5 via
`Seq2SeqTrainer` — this script is the decoder-only counterpart.

Why a separate script
---------------------
`Seq2SeqTrainer` assumes encoder-decoder models and consumes `labels` of
target-only token IDs. Causal LMs need a single concatenated sequence
(`prompt + target`) with labels masked to `-100` over the prompt portion so
loss lands only on the target tokens. Different plumbing, different metrics
surface (no `predict_with_generate` on `Trainer`), so it's cleaner to fork
than branch.

Train-/serve-parity
-------------------
The prompt fed to the model at training time must be byte-identical to the
one `HFCausalGenerate` builds at serve time. This script reuses:
  - `DECOMPOSER_ROUTER_SYSTEM_PROMPT` from `council_policies.p4.hf_causal_generate`
  - `INPUT_PREFIX` + `featurize(question, context)` from the router pipeline
  - `serialize_targets` from `seq2seq_decomposer_router` so labels match the
    parser used in `parse_targets` at eval and serve time.

Label masking
-------------
Standard causal-LM SFT: tokenize the chat-wrapped prompt and the target
separately, concat, set labels to -100 over the prompt. This keeps gradient
signal on the `role: subtask` tokens (the thing we want the model to learn)
and not on the ~400-token system prompt we've already put there by hand.

LoRA
----
Attention-only (q/k/v/o_proj), r=16, alpha=32 — chosen to add capacity
without overfitting the ~4.5k-row dataset. Adapters save to
``<out>/adapter/`` and load via `PeftModel.from_pretrained` in serving.

Usage
-----
    uv run --package training --extra decomposer-router-causal \\
      python -m training.train_decomposer_router_causal \\
      --dataset-dir artifacts/decomposer_router_dataset/parquet \\
      --out artifacts/decomposer_router_causal
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "google/gemma-2-2b-it"
DEFAULT_SEED = 42


@dataclass(slots=True)
class TrainArgs:
    out: Path
    dataset_dir: Path
    base_model: str
    epochs: int
    lr: float
    batch_size: int
    grad_accum_steps: int
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    warmup_ratio: float
    weight_decay: float
    seed: int


def parse_args(argv: list[str] | None = None) -> TrainArgs:
    p = argparse.ArgumentParser(
        description="LoRA-fine-tune a causal chat LM into the joint "
        "decomposer+router.",
    )
    p.add_argument("--out", type=Path, required=True,
                   help="Artifact directory (contains adapter/, card.json).")
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="Directory with train.parquet, dev.parquet, "
                        "mini_test.parquet, gold_eval.parquet.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ns = p.parse_args(argv)
    return TrainArgs(
        out=ns.out,
        dataset_dir=ns.dataset_dir,
        base_model=ns.base_model,
        epochs=ns.epochs,
        lr=ns.lr,
        batch_size=ns.batch_size,
        grad_accum_steps=ns.grad_accum_steps,
        max_seq_length=ns.max_seq_length,
        lora_r=ns.lora_r,
        lora_alpha=ns.lora_alpha,
        lora_dropout=ns.lora_dropout,
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


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_prompt_and_target(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    system_prompt: str,
    input_prefix: str,
    featurize_fn: Any,
    serialize_targets_fn: Any,
) -> tuple[str, str]:
    """Return (chat_prompt, target) matching `HFCausalGenerate` at serve time.

    The chat_prompt includes the chat-template wrapping plus the generation
    prompt marker; target is the raw newline-joined `role: subtask` block
    plus an EOS so the model learns to stop.
    """
    featurized = input_prefix + featurize_fn(
        row.get("question") or "",
        row.get("context") or "",
    )
    user_msg = f"{system_prompt}\n\nInput: {featurized}\nOutput:"
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    target = serialize_targets_fn(row.get("targets") or [])
    return chat_prompt, target


def _make_tokenize_fn(
    tokenizer: Any,
    *,
    max_seq_length: int,
    system_prompt: str,
    input_prefix: str,
    featurize_fn: Any,
    serialize_targets_fn: Any,
):
    """Return a per-row tokenizer that produces {input_ids, attention_mask, labels}.

    `labels` are -100 over the prompt tokens and target-token-ids over the
    target portion. When the combined sequence exceeds `max_seq_length`, we
    truncate from the left of the prompt — featurized context is less
    critical than preserving the target, and `featurize` already caps context
    to ~4k chars so overflow is rare.
    """
    eos_token = tokenizer.eos_token or "</s>"

    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        chat_prompt, target = _build_prompt_and_target(
            row,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            input_prefix=input_prefix,
            featurize_fn=featurize_fn,
            serialize_targets_fn=serialize_targets_fn,
        )
        prompt_ids = tokenizer(
            chat_prompt, add_special_tokens=False,
        )["input_ids"]
        target_ids = tokenizer(
            target + eos_token, add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)

        if len(input_ids) > max_seq_length:
            overflow = len(input_ids) - max_seq_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return tokenize


class _CausalCollator:
    """Pad {input_ids, attention_mask, labels} to longest in batch.

    HF's `DataCollatorForLanguageModeling` can do this but it assumes MLM
    semantics and re-pads labels with the pad-id rather than -100. Rolling
    our own is ~15 lines and avoids surprises.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    _seed_everything(args.seed)

    # Heavy deps imported here so `--help` and module-grep stay cheap.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: F401
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from council_policies.p4.hf_causal_generate import (
        DECOMPOSER_ROUTER_SYSTEM_PROMPT,
    )
    from council_policies.p4.router_featurize import featurize
    from council_policies.p4.seq2seq_decomposer_router import (
        INPUT_PREFIX,
        serialize_targets,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("loading base model %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    # Gemma-2 requires eager attention for correct logits during training;
    # flash-attn and sdpa both subtly differ from the published numerics.

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_files = {
        "train": str(args.dataset_dir / "train.parquet"),
        "validation": str(args.dataset_dir / "dev.parquet"),
    }
    ds = load_dataset("parquet", data_files=data_files)

    tokenize_fn = _make_tokenize_fn(
        tokenizer,
        max_seq_length=args.max_seq_length,
        system_prompt=DECOMPOSER_ROUTER_SYSTEM_PROMPT,
        input_prefix=INPUT_PREFIX,
        featurize_fn=featurize,
        serialize_targets_fn=serialize_targets,
    )
    # `remove_columns` drops the source fields — Trainer will refuse if we
    # leave them on since they aren't tensors.
    remove_cols = list(ds["train"].column_names)
    tokenized = ds.map(
        tokenize_fn, remove_columns=remove_cols, batched=False,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        bf16=True,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=_CausalCollator(tokenizer.pad_token_id),
    )

    logger.info("starting training: %d train / %d eval rows",
                len(tokenized["train"]), len(tokenized["validation"]))
    trainer.train()

    # Save LoRA adapter separately so the serving path can load it via
    # `PeftModel.from_pretrained` on top of the base model without needing
    # a full-weight checkpoint.
    adapter_dir = args.out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    card = {
        "base_model": args.base_model,
        "adapter_dir": str(adapter_dir),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "hparams": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
        },
        "dataset_dir": str(args.dataset_dir),
        "n_train": len(tokenized["train"]),
        "n_eval": len(tokenized["validation"]),
    }
    (args.out / "card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8",
    )
    logger.info("wrote adapter → %s and card → %s",
                adapter_dir, args.out / "card.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
