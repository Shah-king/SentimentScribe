"""
Data and prediction drift detection using Evidently AI.
Generates HTML reports suitable for CI/CD artifact storage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _check_evidently() -> None:
    try:
        import evidently  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "evidently is required for drift detection. "
            "Install with: pip install evidently"
        ) from exc


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    text_column: str = "review",
    output_path: str | Path = "reports/data_drift.html",
) -> None:
    """Compare reference and current datasets for text drift.

    Generates an interactive HTML report using Evidently.

    Parameters
    ----------
    reference_df:
        Training / baseline DataFrame with ``text_column`` and ``label``.
    current_df:
        New / production DataFrame to compare against.
    text_column:
        Name of the text feature column.
    output_path:
        Path to write the HTML report.
    """
    _check_evidently()
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.pipeline.column_mapping import ColumnMapping

    # Evidently requires numeric features; use text length as a proxy
    for df in (reference_df, current_df):
        df["text_length"] = df[text_column].str.split().str.len()
        df["char_length"] = df[text_column].str.len()

    column_mapping = ColumnMapping(
        target="label",
        numerical_features=["text_length", "char_length"],
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info("Data drift report saved → %s", output_path)


def detect_prediction_drift(
    reference_predictions: pd.DataFrame,
    current_predictions: pd.DataFrame,
    output_path: str | Path = "reports/prediction_drift.html",
) -> None:
    """Detect drift in model predictions over time.

    Parameters
    ----------
    reference_predictions:
        DataFrame with ``prediction`` (0/1) and ``confidence`` columns from baseline.
    current_predictions:
        DataFrame with same schema from current production traffic.
    output_path:
        Path to write the HTML report.
    """
    _check_evidently()
    from evidently.report import Report
    from evidently.metric_preset import TargetDriftPreset
    from evidently.pipeline.column_mapping import ColumnMapping

    column_mapping = ColumnMapping(
        prediction="prediction",
        numerical_features=["confidence"],
    )

    report = Report(metrics=[TargetDriftPreset()])
    report.run(
        reference_data=reference_predictions,
        current_data=current_predictions,
        column_mapping=column_mapping,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info("Prediction drift report saved → %s", output_path)
