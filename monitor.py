"""
SentimentScribe — Monitoring entry point.

Compares a new batch of data/predictions against the training baseline
and generates Evidently AI HTML drift reports.

Usage
-----
# Detect data drift
python monitor.py --type data --current data/new_reviews.csv

# Detect prediction drift
python monitor.py --type predictions --current data/new_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.data_pipeline.load_data import load_raw, clean_and_label
from src.monitoring.drift_detector import detect_data_drift, detect_prediction_drift
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/monitor.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentimentScribe drift monitoring")
    parser.add_argument(
        "--type",
        choices=["data", "predictions"],
        default="data",
        help="Type of drift to detect",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current data CSV to compare against reference",
    )
    parser.add_argument(
        "--reference",
        default="data/bookReviewsData.csv",
        help="Reference dataset path (default: training data)",
    )
    parser.add_argument(
        "--output",
        default="reports/drift_report.html",
        help="Output path for the HTML report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading reference data from %s…", args.reference)
    reference_df = clean_and_label(load_raw(args.reference))

    logger.info("Loading current data from %s…", args.current)
    current_df = pd.read_csv(args.current)

    if args.type == "data":
        if "review" not in current_df.columns:
            # Try auto-clean
            current_df = clean_and_label(current_df)
        detect_data_drift(
            reference_df=reference_df,
            current_df=current_df,
            output_path=args.output,
        )
    elif args.type == "predictions":
        required = {"prediction", "confidence"}
        if not required.issubset(current_df.columns):
            logger.error(
                "Predictions CSV must have columns %s. Got: %s",
                required,
                current_df.columns.tolist(),
            )
            sys.exit(1)

        # Build reference predictions from training data
        from src.inference.predictor import get_predictor
        predictor = get_predictor()
        sentiments, confidences = zip(
            *[predictor.predict(r) for r in reference_df["review"].tolist()]
        )
        ref_pred_df = pd.DataFrame({
            "prediction": [1 if s == "positive" else 0 for s in sentiments],
            "confidence": list(confidences),
        })

        detect_prediction_drift(
            reference_predictions=ref_pred_df,
            current_predictions=current_df,
            output_path=args.output,
        )

    logger.info("Report saved → %s", args.output)


if __name__ == "__main__":
    main()
