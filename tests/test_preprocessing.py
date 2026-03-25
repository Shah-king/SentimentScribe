"""Tests for the data loading and text preprocessing modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data_pipeline.load_data import clean_and_label, _clean_text
from src.features.text_preprocessor import TextPreprocessor


# ── clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_lowercase():
    assert _clean_text("Hello World") == "hello world"


def test_clean_text_removes_punctuation():
    result = _clean_text("Great book! Really, truly great.")
    assert "!" not in result
    assert "," not in result
    assert "." not in result


def test_clean_text_handles_empty():
    assert _clean_text("") == ""


def test_clean_text_handles_numbers():
    result = _clean_text("Rating: 5/5")
    assert result == "rating 55"


# ── clean_and_label ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Review": [
            "This book was absolutely wonderful and moving.",
            "Terrible waste of time. The worst book I've read.",
            "A decent read overall, nothing special.",
        ],
        "Positive Review": [True, False, True],
    })


def test_clean_and_label_columns(sample_raw_df):
    result = clean_and_label(sample_raw_df)
    assert "review" in result.columns
    assert "label" in result.columns


def test_clean_and_label_positive_is_1(sample_raw_df):
    result = clean_and_label(sample_raw_df)
    assert result.iloc[0]["label"] == 1


def test_clean_and_label_negative_is_0(sample_raw_df):
    result = clean_and_label(sample_raw_df)
    assert result.iloc[1]["label"] == 0


def test_clean_and_label_drops_missing():
    df = pd.DataFrame({
        "Review": [None, "Good book"],
        "Positive Review": [True, True],
    })
    result = clean_and_label(df)
    assert len(result) == 1


def test_clean_and_label_max_words(sample_raw_df):
    result = clean_and_label(sample_raw_df, max_words=3)
    for review in result["review"]:
        assert len(review.split()) <= 3


def test_clean_and_label_raises_on_wrong_columns():
    df = pd.DataFrame({"text": ["hello"], "rating": [5]})
    with pytest.raises(ValueError, match="Expected columns"):
        clean_and_label(df)


# ── TextPreprocessor ──────────────────────────────────────────────────────────

SAMPLE_CORPUS = [
    "this book was absolutely wonderful",
    "terrible waste of time worst book",
    "great characters and compelling plot",
    "boring and disappointing nothing works",
    "loved every page could not put it down",
]


def test_preprocessor_fit_transform_shape():
    # min_df=1 because the sample corpus is tiny (5 docs)
    tp = TextPreprocessor(max_features=100, min_df=1)
    X = tp.fit_transform(SAMPLE_CORPUS)
    assert X.shape[0] == len(SAMPLE_CORPUS)


def test_preprocessor_transform_without_fit_raises():
    tp = TextPreprocessor()
    with pytest.raises(RuntimeError, match="Call .fit\\(\\) before"):
        tp.transform(SAMPLE_CORPUS)


def test_preprocessor_save_load(tmp_path):
    tp = TextPreprocessor(max_features=100, min_df=1)
    tp.fit(SAMPLE_CORPUS)
    save_path = tmp_path / "vectorizer.pkl"
    tp.save(save_path)

    loaded = TextPreprocessor.load(save_path)
    original_vec = tp.transform(SAMPLE_CORPUS).toarray()
    loaded_vec = loaded.transform(SAMPLE_CORPUS).toarray()
    import numpy as np
    assert np.allclose(original_vec, loaded_vec)


def test_preprocessor_feature_names():
    tp = TextPreprocessor(max_features=100, min_df=1)
    tp.fit(SAMPLE_CORPUS)
    names = tp.get_feature_names()
    assert isinstance(names, list)
    assert len(names) > 0
