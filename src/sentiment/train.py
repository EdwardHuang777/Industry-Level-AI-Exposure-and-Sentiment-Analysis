"""Fine-tune `roberta-base` on weakly-labeled entity-context data.

4-class sequence classification: negative / neutral / positive / unclear.
Class-weighted CrossEntropyLoss (computed via sklearn balanced weights)
addresses minority-class imbalance, especially for `unclear` and `negative`.
"""
from __future__ import annotations

from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import (
    RANDOM_SEED,
    SENTIMENT_DIR,
    SENTIMENT_LABEL_TO_ID,
    SENTIMENT_LABELS,
    SENTIMENT_MODEL_DIR,
)


# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_MODEL = "roberta-base"
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
MAX_LENGTH = 256
MODEL_OUT = SENTIMENT_MODEL_DIR / "roberta-base-news-sentiment"


# ── Class-weighted Trainer ────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """Trainer that applies a fixed per-class weight to the CE loss."""

    def __init__(self, *args, class_weights: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        weight = torch.tensor(self._class_weights, device=logits.device)
        loss = torch.nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Metrics ───────────────────────────────────────────────────────────────────
def _build_metrics():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }
    return compute_metrics


# ── Main entry ────────────────────────────────────────────────────────────────
def train(labeled_df: pd.DataFrame | None = None,
          out_dir: Path = MODEL_OUT,
          base_model: str = BASE_MODEL,
          num_epochs: int = NUM_EPOCHS,
          learning_rate: float = LEARNING_RATE,
          weight_decay: float = WEIGHT_DECAY,
          train_batch_size: int = TRAIN_BATCH_SIZE,
          eval_batch_size: int = EVAL_BATCH_SIZE,
          max_length: int = MAX_LENGTH,
          seed: int = RANDOM_SEED) -> dict:
    """Fine-tune RoBERTa. Returns the val-set metrics dict."""
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    if labeled_df is None:
        labeled_df = pd.read_parquet(SENTIMENT_DIR / "labeled_org_entity_context.parquet")

    df = labeled_df.copy()
    df["labels"] = df["labels"].map(SENTIMENT_LABEL_TO_ID).astype("int64")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["labels"],
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    def tokenize(batch):
        return tokenizer(batch["org_entity_text"], truncation=True, max_length=max_length)

    hf_train = Dataset.from_pandas(
        train_df[["org_entity_text", "labels"]], preserve_index=False,
    ).map(tokenize, batched=True, remove_columns=["org_entity_text"])
    hf_val = Dataset.from_pandas(
        val_df[["org_entity_text", "labels"]], preserve_index=False,
    ).map(tokenize, batched=True, remove_columns=["org_entity_text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(SENTIMENT_LABELS),
        id2label=SENTIMENT_LABELS,
        label2id=SENTIMENT_LABEL_TO_ID,
    )

    classes = np.arange(len(SENTIMENT_LABELS))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_df["labels"].values,
    ).astype(np.float32)

    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_build_metrics(),
        class_weights=weights,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved fine-tuned model → {out_dir}")

    # Detailed validation report
    pred = trainer.predict(hf_val)
    y_pred = np.argmax(pred.predictions, axis=-1)
    target_names = [SENTIMENT_LABELS[i] for i in sorted(SENTIMENT_LABELS)]
    print(classification_report(pred.label_ids, y_pred, target_names=target_names, digits=4))
    cm = confusion_matrix(pred.label_ids, y_pred, labels=list(sorted(SENTIMENT_LABELS)))
    print(pd.DataFrame(cm, index=target_names, columns=target_names))
    return metrics
