"""Tests for sklearn baseline model wrappers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.tfidf_models import LogisticRegressionModel, NaiveBayesModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_data():
    """Simple 2D sparse feature matrix + binary labels for quick training."""
    X = csr_matrix(np.array([
        [0.5, 0.1, 0.0, 0.8],
        [0.0, 0.9, 0.7, 0.1],
        [0.6, 0.2, 0.1, 0.7],
        [0.1, 0.8, 0.9, 0.0],
        [0.7, 0.1, 0.0, 0.9],
        [0.0, 0.7, 0.8, 0.2],
    ]))
    y = np.array([1, 0, 1, 0, 1, 0])
    return X, y


# ── LogisticRegressionModel ───────────────────────────────────────────────────

class TestLogisticRegressionModel:

    def test_fit_predict(self, binary_data):
        X, y = binary_data
        model = LogisticRegressionModel(max_iter=200)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (6,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, binary_data):
        X, y = binary_data
        model = LogisticRegressionModel(max_iter=200)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (6, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_evaluate_returns_required_keys(self, binary_data):
        X, y = binary_data
        model = LogisticRegressionModel(max_iter=200)
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        for key in ("accuracy", "roc_auc", "confusion_matrix", "classification_report"):
            assert key in metrics

    def test_accuracy_in_range(self, binary_data):
        X, y = binary_data
        model = LogisticRegressionModel(max_iter=200)
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_load(self, binary_data, tmp_path):
        X, y = binary_data
        model = LogisticRegressionModel(max_iter=200)
        model.fit(X, y)
        preds_before = model.predict(X)

        save_path = tmp_path / "lr.pkl"
        model.save(save_path)
        loaded = LogisticRegressionModel.load(save_path)
        preds_after = loaded.predict(X)

        assert np.array_equal(preds_before, preds_after)


# ── NaiveBayesModel ───────────────────────────────────────────────────────────

class TestNaiveBayesModel:

    def test_fit_predict(self, binary_data):
        X, y = binary_data
        # NB requires non-negative features (already satisfied)
        model = NaiveBayesModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (6,)

    def test_predict_proba_sums_to_one(self, binary_data):
        X, y = binary_data
        model = NaiveBayesModel()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_load(self, binary_data, tmp_path):
        X, y = binary_data
        model = NaiveBayesModel()
        model.fit(X, y)
        preds_before = model.predict(X)

        save_path = tmp_path / "nb.pkl"
        model.save(save_path)
        loaded = NaiveBayesModel.load(save_path)
        preds_after = loaded.predict(X)

        assert np.array_equal(preds_before, preds_after)
