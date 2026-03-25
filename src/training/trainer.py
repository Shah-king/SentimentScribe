"""
Production training pipeline.
Loads data, trains baseline and/or transformer models,
logs everything to MLflow, and saves versioned artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import yaml

from src.data_pipeline.load_data import load_dataset
from src.features.text_preprocessor import TextPreprocessor
from src.models.tfidf_models import LogisticRegressionModel, NaiveBayesModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

ModelType = Literal["logistic_regression", "naive_bayes", "distilbert"]

ARTIFACT_DIR = Path("experiments/artifacts")


def _load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _save_confusion_matrix_plot(
    cm: list[list[int]],
    output_path: str | Path,
) -> None:
    """Save confusion matrix as PNG for MLflow artifact logging."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_roc_plot(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_baseline(
    config: Dict[str, Any],
    model_type: ModelType = "logistic_regression",
    experiment_name: str = "sentimentscribe-baseline",
) -> Dict[str, Any]:
    """Train a TF-IDF + sklearn baseline model with MLflow tracking.

    Parameters
    ----------
    config:
        Loaded YAML config dict.
    model_type:
        One of ``logistic_regression`` or ``naive_bayes``.
    experiment_name:
        MLflow experiment name.

    Returns
    -------
    dict
        Evaluation metrics from the run.
    """
    data_cfg = config.get("data", {})
    model_cfg = config.get("models", {}).get(model_type, {})
    tfidf_cfg = config.get("tfidf", {})

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading dataset…")
    X_train, X_test, y_train, y_test = load_dataset(
        max_words=data_cfg.get("max_words", 500),
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
    )

    # ── Feature engineering ──────────────────────────────────────────────────
    preprocessor = TextPreprocessor(
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        max_features=tfidf_cfg.get("max_features", 50000),
    )
    X_train_vec = preprocessor.fit_transform(X_train)
    X_test_vec = preprocessor.transform(X_test)

    # ── Model ────────────────────────────────────────────────────────────────
    if model_type == "logistic_regression":
        model = LogisticRegressionModel(**model_cfg)
    elif model_type == "naive_bayes":
        model = NaiveBayesModel(**model_cfg)
    else:
        raise ValueError(f"Unsupported baseline model: {model_type}")

    # ── MLflow run ───────────────────────────────────────────────────────────
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_type) as run:
        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)

        # Log params
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("ngram_range", tfidf_cfg.get("ngram_range", [1, 2]))
        mlflow.log_param("max_features", tfidf_cfg.get("max_features", 50000))
        mlflow.log_params(model_cfg)

        # Train
        model.fit(X_train_vec, y_train.to_numpy())

        # Evaluate
        metrics = model.evaluate(X_test_vec, y_test.to_numpy())
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])

        logger.info("\n%s", metrics["classification_report"])

        # Confusion matrix plot
        cm_path = ARTIFACT_DIR / run_id / "confusion_matrix.png"
        _save_confusion_matrix_plot(metrics["confusion_matrix"], cm_path)
        mlflow.log_artifact(str(cm_path))

        # ROC curve plot
        y_prob = model.predict_proba(X_test_vec)[:, 1]
        roc_path = ARTIFACT_DIR / run_id / "roc_curve.png"
        _save_roc_plot(y_test.to_numpy(), y_prob, roc_path)
        mlflow.log_artifact(str(roc_path))

        # Save model + vectorizer artifacts
        model_path = ARTIFACT_DIR / run_id / f"{model_type}.pkl"
        vec_path = ARTIFACT_DIR / run_id / "vectorizer.pkl"
        model.save(model_path)
        preprocessor.save(vec_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(vec_path))

        mlflow.sklearn.log_model(model._model, artifact_path="sklearn_model")

    return metrics


def train_transformer(
    config: Dict[str, Any],
    experiment_name: str = "sentimentscribe-distilbert",
) -> Dict[str, Any]:
    """Fine-tune DistilBERT and log to MLflow."""
    from src.models.transformer_model import DistilBERTSentimentModel

    data_cfg = config.get("data", {})
    bert_cfg = config.get("models", {}).get("distilbert", {})

    logger.info("Loading dataset for DistilBERT…")
    X_train, X_test, y_train, y_test = load_dataset(
        max_words=data_cfg.get("max_words", 500),
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
    )

    bert_model = DistilBERTSentimentModel(
        max_length=bert_cfg.get("max_length", 256)
    )

    output_dir = Path("experiments/distilbert")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="distilbert") as run:
        run_id = run.info.run_id
        mlflow.log_params(bert_cfg)

        metrics = bert_model.train(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            y_eval=y_test,
            output_dir=output_dir,
            num_train_epochs=bert_cfg.get("num_epochs", 3),
            per_device_train_batch_size=bert_cfg.get("batch_size", 16),
            learning_rate=bert_cfg.get("learning_rate", 2e-5),
        )

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k.replace("eval_", ""), v)

        save_path = ARTIFACT_DIR / run_id / "distilbert"
        bert_model.save(save_path)
        mlflow.log_artifacts(str(save_path), artifact_path="distilbert_model")

    return metrics
