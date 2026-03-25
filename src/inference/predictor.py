"""
Inference module — loads the latest versioned model from disk
and serves predictions for the FastAPI layer.
"""

from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from src.features.text_preprocessor import TextPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

ARTIFACTS_DIR = Path("experiments/artifacts")

SentimentLabel = Literal["positive", "negative"]


def _find_latest_run_dir(artifacts_dir: Path) -> Optional[Path]:
    """Return the most-recently-modified run subdirectory."""
    run_dirs = sorted(
        [d for d in artifacts_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


class SentimentPredictor:
    """Load model + vectorizer and run inference.

    Supports both TF-IDF sklearn models and DistilBERT.

    Parameters
    ----------
    model_path:
        Path to a pickled sklearn model (.pkl) or a DistilBERT directory.
    vectorizer_path:
        Path to a pickled TF-IDF vectorizer. Required for sklearn models.
    model_type:
        ``"sklearn"`` or ``"distilbert"``.
    """

    def __init__(
        self,
        model_path: str | Path,
        vectorizer_path: Optional[str | Path] = None,
        model_type: Literal["sklearn", "distilbert"] = "sklearn",
    ) -> None:
        self.model_type = model_type
        self._model = None
        self._vectorizer: Optional[TextPreprocessor] = None

        if model_type == "sklearn":
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            if vectorizer_path:
                self._vectorizer = TextPreprocessor.load(vectorizer_path)
            logger.info("Sklearn model loaded from %s", model_path)
        elif model_type == "distilbert":
            from src.models.transformer_model import DistilBERTSentimentModel
            self._model = DistilBERTSentimentModel.load(model_path)
            logger.info("DistilBERT model loaded from %s", model_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def predict(self, text: str) -> Tuple[SentimentLabel, float]:
        """Predict sentiment for a single review.

        Parameters
        ----------
        text:
            Raw review string.

        Returns
        -------
        label:
            ``"positive"`` or ``"negative"``.
        confidence:
            Probability score for the predicted class (0–1).
        """
        if self.model_type == "sklearn":
            return self._predict_sklearn(text)
        return self._predict_distilbert(text)

    def _predict_sklearn(self, text: str) -> Tuple[SentimentLabel, float]:
        vec = self._vectorizer.transform([text])
        pred = self._model.predict(vec)[0]
        prob = self._model.predict_proba(vec)[0]
        confidence = float(prob[pred])
        label: SentimentLabel = "positive" if pred == 1 else "negative"
        return label, confidence

    def _predict_distilbert(self, text: str) -> Tuple[SentimentLabel, float]:
        labels, confidences = self._model.predict([text])
        label: SentimentLabel = "positive" if labels[0] == 1 else "negative"
        return label, float(confidences[0])


@lru_cache(maxsize=1)
def get_predictor(
    model_type: Literal["sklearn", "distilbert"] = "sklearn",
    artifacts_dir: str = str(ARTIFACTS_DIR),
) -> SentimentPredictor:
    """Cached factory — load once per process.

    Scans ``artifacts_dir`` for the most recent run and loads
    the appropriate model files.
    """
    base = Path(artifacts_dir)
    run_dir = _find_latest_run_dir(base)

    if run_dir is None:
        raise FileNotFoundError(
            f"No trained artifacts found in {base}. Run train.py first."
        )

    logger.info("Loading model from run directory: %s", run_dir)

    if model_type == "distilbert":
        bert_dir = run_dir / "distilbert"
        if not bert_dir.exists():
            raise FileNotFoundError(f"DistilBERT artifacts not found at {bert_dir}")
        return SentimentPredictor(bert_dir, model_type="distilbert")

    # Prefer logistic regression, fall back to naive bayes
    for name in ("logistic_regression.pkl", "naive_bayes.pkl"):
        model_path = run_dir / name
        if model_path.exists():
            vec_path = run_dir / "vectorizer.pkl"
            return SentimentPredictor(model_path, vec_path, model_type="sklearn")

    raise FileNotFoundError(
        f"No sklearn model found in {run_dir}. Run train.py first."
    )
