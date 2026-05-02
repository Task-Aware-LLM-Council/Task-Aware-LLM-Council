"""
Fine-tune distilroberta-base into the P4 learned router.

Run:

    uv run -m training.train_router --out artifacts/router_model

Pipeline:
    1. Load `task-aware-llm-council/router-dataset` (dev + mini_test).
    2. Featurize each row via `council_policies.featurize`
       (same function that ships at inference time — single source of
       truth, see `council_policies/router_featurize.py`).
    3. Collapse `skill_tags` → 3-role label via
       `council_policies.role_from_tags`.
    4. Split dev into train/eval (deterministic, seeded).
    5. Fine-tune `distilroberta-base` with a 3-class head.
    6. Save tokenizer + model to `--out` and a sibling
       `router_card.json` (schema: `council_policies.RouterCard`).
    7. Eval on dev + mini_test, record macro F1 + confusion matrix
       in the card.

Heavy deps (`torch`, `transformers`, `datasets`, `numpy`,
`scikit-learn`) import inside `main()`. This is a CLI entrypoint, so
lazy imports keep the module free to grep/document without pulling
the ML stack into unrelated tooling.
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # noqa: F401

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "task-aware-llm-council/router-dataset"
DEFAULT_BASE_MODEL = "distilroberta-base"
DEFAULT_SEED = 42


@dataclass(slots=True)
class TrainArgs:
    out: Path
    dataset: str
    dataset_revision: str
    base_model: str
    epochs: int
    lr: float
    batch_size: int
    max_length: int
    dev_eval_frac: float
    seed: int
    floor: float


def parse_args(argv: list[str] | None = None) -> TrainArgs:
    p = argparse.ArgumentParser(description="Train the P4 learned router.")
    p.add_argument("--out", type=Path, required=True, help="Artifact directory.")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument(
        "--dataset-revision",
        default="",
        help="HF dataset revision/SHA to pin. Empty = latest.",
    )
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument(
        "--dev-eval-frac",
        type=float,
        default=0.2,
        help="Fraction of `dev` held out for early-stopping eval.",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--floor",
        type=float,
        default=0.3,
        help="Confidence floor written into the router card. "
        "Calibrate from dev ROC in a follow-up pass if desired.",
    )
    ns = p.parse_args(argv)
    return TrainArgs(
        out=ns.out,
        dataset=ns.dataset,
        dataset_revision=ns.dataset_revision,
        base_model=ns.base_model,
        epochs=ns.epochs,
        lr=ns.lr,
        batch_size=ns.batch_size,
        max_length=ns.max_length,
        dev_eval_frac=ns.dev_eval_frac,
        seed=ns.seed,
        floor=ns.floor,
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


def _rows_to_examples(rows, featurize, role_from_tags, role_to_index):
    """Map raw HF rows → `(text, label)` tuples. Unknown-tag rows fall
    back to the generalist label; skipping them silently would bias
    the label distribution."""
    texts: list[str] = []
    labels: list[int] = []
    for row in rows:
        text = featurize(row["question"], row.get("context", "") or "")
        role = role_from_tags(row.get("skill_tags") or [])
        texts.append(text)
        labels.append(role_to_index(role))
    return texts, labels


def _compute_metrics_fn(role_labels: tuple[str, ...]):
    """Return the `compute_metrics` callback closed over the label
    space. `Trainer` calls this with a `(predictions, labels)` pair."""
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        cm = confusion_matrix(
            labels, preds, labels=list(range(len(role_labels)))
        ).tolist()
        return {"macro_f1": float(macro_f1), "confusion": cm}

    return compute_metrics


def _load_raw_rows(dataset_name: str, revision: str):
    from datasets import load_dataset

    kwargs = {"revision": revision} if revision else {}
    ds = load_dataset(dataset_name, **kwargs)
    if "dev" not in ds or "mini_test" not in ds:
        raise KeyError(
            f"Router dataset {dataset_name!r} missing required splits "
            f"('dev', 'mini_test'); got {list(ds.keys())}"
        )
    return list(ds["dev"]), list(ds["mini_test"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    _seed_everything(args.seed)

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    from council_policies import RouterCard
    from council_policies.p4.router_featurize import (
        DEFAULT_CONTEXT_CHAR_CAP,
        featurize,
    )
    from council_policies.p4.router_labels import (
        DEFAULT_FALLBACK_ROLE,
        ROLE_LABELS,
        role_from_tags,
        role_to_index,
    )

    dev_rows, mini_test_rows = _load_raw_rows(
        args.dataset, args.dataset_revision
    )

    # Deterministic train/eval split of the dev pool. `mini_test` stays
    # fully held out for post-training eval; we never train or
    # early-stop on it.
    rng = random.Random(args.seed)
    dev_rows = dev_rows[:]
    rng.shuffle(dev_rows)
    eval_size = max(1, int(len(dev_rows) * args.dev_eval_frac))
    train_rows = dev_rows[eval_size:]
    eval_rows = dev_rows[:eval_size]

    train_texts, train_labels = _rows_to_examples(
        train_rows, featurize, role_from_tags, role_to_index
    )
    eval_texts, eval_labels = _rows_to_examples(
        eval_rows, featurize, role_from_tags, role_to_index
    )
    mini_texts, mini_labels = _rows_to_examples(
        mini_test_rows, featurize, role_from_tags, role_to_index
    )

    logger.info(
        "Dataset sizes — train=%d eval=%d mini_test=%d",
        len(train_texts),
        len(eval_texts),
        len(mini_texts),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def _tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=args.max_length
        )

    def _to_hf(texts, labels):
        return (
            Dataset.from_dict({"text": texts, "label": labels})
            .map(_tokenize, batched=True)
            .remove_columns(["text"])
        )

    train_ds = _to_hf(train_texts, train_labels)
    eval_ds = _to_hf(eval_texts, eval_labels)
    mini_ds = _to_hf(mini_texts, mini_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(ROLE_LABELS),
        id2label={i: r for i, r in enumerate(ROLE_LABELS)},
        label2id={r: i for i, r in enumerate(ROLE_LABELS)},
    )

    training_args = TrainingArguments(
        output_dir=str(args.out / "_trainer_state"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics_fn(ROLE_LABELS),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    dev_metrics = trainer.evaluate(eval_ds)
    mini_metrics = trainer.evaluate(mini_ds)

    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    card = RouterCard(
        model_version=_model_version(args.base_model),
        roles=ROLE_LABELS,
        floor=args.floor,
        fallback_role=DEFAULT_FALLBACK_ROLE,
        base_model=args.base_model,
        context_char_cap=DEFAULT_CONTEXT_CHAR_CAP,
        dataset_revision=args.dataset_revision,
        metrics={
            "dev_macro_f1": float(dev_metrics.get("eval_macro_f1", 0.0)),
            "dev_confusion": dev_metrics.get("eval_confusion", []),
            "mini_test_macro_f1": float(
                mini_metrics.get("eval_macro_f1", 0.0)
            ),
            "mini_test_confusion": mini_metrics.get("eval_confusion", []),
            "n_train": len(train_texts),
            "n_eval": len(eval_texts),
            "n_mini_test": len(mini_texts),
        },
        training_config={
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "dev_eval_frac": args.dev_eval_frac,
            "seed": args.seed,
        },
        git_sha=_git_sha(),
    )
    card_path = args.out / "router_card.json"
    card.save(card_path)

    logger.info("Saved router artifact to %s", args.out)
    logger.info("Card: %s", card_path)
    print(json.dumps(card.metrics, indent=2))

    # Silence unused-import noise when running under strict linters.
    _ = (np, torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
