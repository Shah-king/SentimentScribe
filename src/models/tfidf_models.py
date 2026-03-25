"""
Scikit-learn baseline models: Logistic Regression and Naive Bayes.
Each model is wrapped in a thin class for consistent save/load and
MLflow-compatible evaluation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class BaseModel:
    """Shared interface for sklearn sentiment classifiers."""

    name: str = "base"
    _model: Any = None

    def fit(self, X_train: spmatrix, y_train: np.ndarray) -> "BaseModel":
        logger.info("Training %s…", self.name)
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X: spmatrix) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: spmatrix) -> np.ndarray:
        return self._model.predict_proba(X)

    def evaluate(
        self, X_test: spmatrix, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Return a dict of evaluation metrics."""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["Negative", "Positive"]
            ),
        }
        logger.info(
            "%s → accuracy=%.4f, AUC=%.4f",
            self.name,
            metrics["accuracy"],
            metrics["roc_auc"],
        )
        return metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("%s saved → %s", self.name, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        obj = cls.__new__(cls)
        obj._model = model
        logger.info("%s loaded ← %s", cls.name, path)
        return obj


class LogisticRegressionModel(BaseModel):
    """Logistic Regression sentiment classifier."""

    name = "logistic_regression"

    def __init__(self, max_iter: int = 1000, C: float = 1.0, **kwargs) -> None:
        self._model = LogisticRegression(
            max_iter=max_iter, C=C, solver="lbfgs", **kwargs
        )


class NaiveBayesModel(BaseModel):
    """Multinomial Naive Bayes sentiment classifier."""

    name = "naive_bayes"

    def __init__(self, alpha: float = 1.0, **kwargs) -> None:
        self._model = MultinomialNB(alpha=alpha, **kwargs)
