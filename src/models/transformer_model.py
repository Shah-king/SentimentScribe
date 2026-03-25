"""
DistilBERT fine-tuning for binary sentiment classification.
Uses Hugging Face Transformers + Trainer API.
GPU-compatible and config-driven.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports so the module is importable even without transformers installed
try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from sklearn.metrics import accuracy_score, roc_auc_score

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers / torch not installed. DistilBERT model unavailable. "
        "Run: pip install transformers torch datasets"
    )


MODEL_CHECKPOINT = "distilbert-base-uncased"
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}


def _check_deps() -> None:
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers, torch, and datasets are required for DistilBERT. "
            "Install them with: pip install transformers torch datasets"
        )


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


class DistilBERTSentimentModel:
    """Fine-tuned DistilBERT for book-review sentiment.

    Parameters
    ----------
    checkpoint:
        Hugging Face model identifier.
    max_length:
        Maximum token sequence length.
    """

    def __init__(
        self,
        checkpoint: str = MODEL_CHECKPOINT,
        max_length: int = 256,
    ) -> None:
        _check_deps()
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None

    def _load_base_model(self) -> None:
        logger.info("Loading tokenizer and model from '%s'…", self.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

    def _tokenize(self, examples: dict) -> dict:
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
        )

    def _build_hf_dataset(
        self, texts: list[str], labels: list[int]
    ) -> "Dataset":
        ds = Dataset.from_dict({"text": texts, "label": labels})
        ds = ds.map(self._tokenize, batched=True)
        ds = ds.remove_columns(["text"])
        ds.set_format("torch")
        return ds

    def train(
        self,
        X_train: "pd.Series | list[str]",
        y_train: "pd.Series | list[int]",
        X_eval: "pd.Series | list[str]",
        y_eval: "pd.Series | list[int]",
        output_dir: str | Path = "experiments/distilbert",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 32,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        **trainer_kwargs,
    ) -> Dict[str, Any]:
        """Fine-tune DistilBERT and return evaluation metrics."""
        _check_deps()
        self._load_base_model()

        train_ds = self._build_hf_dataset(list(X_train), list(y_train))
        eval_ds = self._build_hf_dataset(list(X_eval), list(y_eval))

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            report_to="none",
            **trainer_kwargs,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )

        logger.info("Starting DistilBERT fine-tuning…")
        trainer.train()

        metrics = trainer.evaluate()
        logger.info("Eval metrics: %s", metrics)
        return metrics

    def predict(self, texts: list[str]) -> tuple[list[int], list[float]]:
        """Run inference on a list of texts.

        Returns
        -------
        labels:
            Predicted class indices (0=negative, 1=positive).
        probabilities:
            Confidence scores for the positive class.
        """
        _check_deps()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        self.model.eval()
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        labels = probs.argmax(axis=-1).tolist()
        confidences = probs[:, 1].tolist()
        return labels, confidences

    def save(self, path: str | Path) -> None:
        """Save model and tokenizer to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("DistilBERT model saved → %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        max_length: int = 256,
    ) -> "DistilBERTSentimentModel":
        """Load a fine-tuned model from disk."""
        _check_deps()
        path = Path(path)
        obj = cls.__new__(cls)
        obj.checkpoint = str(path)
        obj.max_length = max_length
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.model.eval()
        logger.info("DistilBERT model loaded ← %s", path)
        return obj
