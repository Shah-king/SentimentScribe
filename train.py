"""
SentimentScribe — Training entry point.

Usage
-----
# Train logistic regression (default)
python train.py

# Train naive bayes
python train.py --model naive_bayes

# Fine-tune DistilBERT
python train.py --model distilbert

# Custom config
python train.py --config configs/config.yaml --model logistic_regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so `src` imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import train_baseline, train_transformer, _load_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/train.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SentimentScribe training pipeline"
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "naive_bayes", "distilbert"],
        default="logistic_regression",
        help="Model to train (default: logistic_regression)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment name (overrides config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)

    mlflow_cfg = config.get("mlflow", {})

    if args.model == "distilbert":
        exp_name = args.experiment or mlflow_cfg.get(
            "experiment_name_transformer", "sentimentscribe-distilbert"
        )
        logger.info("Starting DistilBERT fine-tuning experiment: %s", exp_name)
        metrics = train_transformer(config, experiment_name=exp_name)
    else:
        exp_name = args.experiment or mlflow_cfg.get(
            "experiment_name_baseline", "sentimentscribe-baseline"
        )
        logger.info(
            "Starting baseline training: model=%s, experiment=%s",
            args.model,
            exp_name,
        )
        metrics = train_baseline(config, model_type=args.model, experiment_name=exp_name)

    logger.info("Training complete.")
    logger.info("Accuracy : %.4f", metrics.get("accuracy", 0))
    logger.info("ROC-AUC  : %.4f", metrics.get("roc_auc", 0))


if __name__ == "__main__":
    main()
