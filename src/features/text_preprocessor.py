"""
TF-IDF feature engineering pipeline.
Wraps sklearn's TfidfVectorizer with project-standard defaults
and adds convenience helpers for fitting, transforming, and persistence.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

VECTORIZER_DEFAULTS = dict(
    stop_words="english",
    max_df=0.9,
    min_df=5,
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=50_000,
)


class TextPreprocessor:
    """Fit/transform text using TF-IDF with project-standard settings.

    Parameters
    ----------
    ngram_range:
        Range of n-gram sizes to extract.
    max_features:
        Vocabulary size cap.
    kwargs:
        Additional kwargs forwarded to :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 50_000,
        **kwargs,
    ) -> None:
        params = {**VECTORIZER_DEFAULTS, "ngram_range": ngram_range, "max_features": max_features, **kwargs}
        self.vectorizer = TfidfVectorizer(**params)
        self._fitted = False

    def fit(self, texts: list[str] | "pd.Series") -> "TextPreprocessor":
        """Fit the vectorizer on training corpus."""
        logger.info("Fitting TF-IDF vectorizer on %d documents…", len(texts))
        self.vectorizer.fit(texts)
        self._fitted = True
        logger.info("Vocabulary size: %d", len(self.vectorizer.vocabulary_))
        return self

    def transform(self, texts: list[str] | "pd.Series") -> spmatrix:
        """Transform texts into TF-IDF feature matrix."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str] | "pd.Series") -> spmatrix:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> list[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    def save(self, path: str | Path) -> None:
        """Persist vectorizer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info("Vectorizer saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TextPreprocessor":
        """Load a previously saved vectorizer."""
        with open(path, "rb") as f:
            vec = pickle.load(f)
        obj = cls.__new__(cls)
        obj.vectorizer = vec
        obj._fitted = True
        logger.info("Vectorizer loaded ← %s", path)
        return obj
