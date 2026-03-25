"""
Data loading module for the Book Review Sentiment Analysis pipeline.
Extracted from Bookreview_Sentiment_Analysis.ipynb and made reproducible.
"""

from __future__ import annotations

import logging
import os
import re
import string
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Project root → data directory
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CSV = DATA_DIR / "bookReviewsData.csv"

# Hugging Face fallback dataset identifier (mirror in case CSV is absent)
HF_DATASET_ID = "mattymchen/book-reviews"


def load_raw(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw book-review CSV into a DataFrame.

    Parameters
    ----------
    path:
        Explicit path to the CSV file.  Defaults to ``data/bookReviewsData.csv``
        inside the project root.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with original column names.

    Raises
    ------
    FileNotFoundError
        If ``path`` is given but does not exist.
    """
    csv_path = Path(path) if path else DEFAULT_CSV

    if not csv_path.exists():
        logger.warning(
            "CSV not found at %s. Attempting to download from Hugging Face…", csv_path
        )
        return _load_from_huggingface()

    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path, header=0)
    logger.info("Loaded %d rows, columns: %s", len(df), df.columns.tolist())
    return df


def _load_from_huggingface() -> pd.DataFrame:
    """Fallback: pull the dataset from Hugging Face Hub."""
    try:
        from datasets import load_dataset  # type: ignore

        logger.info("Downloading '%s' from Hugging Face…", HF_DATASET_ID)
        ds = load_dataset(HF_DATASET_ID, split="train")
        df = ds.to_pandas()
        logger.info("Downloaded %d rows from Hugging Face.", len(df))
        return df
    except Exception as exc:  # pragma: no cover
        raise FileNotFoundError(
            f"bookReviewsData.csv not found and Hugging Face download failed: {exc}. "
            "Place bookReviewsData.csv inside the data/ directory."
        ) from exc


def clean_and_label(df: pd.DataFrame, max_words: int = 500) -> pd.DataFrame:
    """Normalise columns, clean text, and encode the binary label.

    Transformation steps (matching notebook):
    1. Standardise column names to snake_case.
    2. Drop rows with missing review text.
    3. Lowercase and strip punctuation from review text.
    4. Cap review length at ``max_words`` tokens.
    5. Encode ``positive_review`` (bool) → integer label (1 = positive, 0 = negative).

    Parameters
    ----------
    df:
        Raw DataFrame as returned by :func:`load_raw`.
    max_words:
        Maximum word count per review; longer reviews are truncated.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns ``review``, ``label``.
    """
    df = df.copy()

    # Standardise column names
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    )

    required = {"review", "positive_review"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required}, got {df.columns.tolist()}"
        )

    # Drop missing
    before = len(df)
    df = df.dropna(subset=["review", "positive_review"]).reset_index(drop=True)
    logger.info("Dropped %d rows with missing values.", before - len(df))

    # Clean text
    df["review"] = df["review"].apply(_clean_text)

    # Cap length
    df["review"] = df["review"].apply(
        lambda x: " ".join(x.split()[:max_words])
    )

    # Binary label
    df["label"] = df["positive_review"].astype(int)

    return df[["review", "label"]]


def _clean_text(text: str) -> str:
    """Lowercase and remove punctuation from a single review string."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def load_dataset(
    path: Optional[str | Path] = None,
    max_words: int = 500,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """End-to-end loader: raw CSV → cleaned train/test splits.

    Parameters
    ----------
    path:
        Optional path override for the CSV file.
    max_words:
        Maximum words per review (truncation).
    test_size:
        Fraction of data reserved for testing.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Pandas Series ready for scikit-learn pipelines.
    """
    from sklearn.model_selection import train_test_split

    raw = load_raw(path)
    cleaned = clean_and_label(raw, max_words=max_words)

    logger.info(
        "Label distribution:\n%s", cleaned["label"].value_counts().to_string()
    )

    X = cleaned["review"]
    y = cleaned["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        "Split → train: %d, test: %d", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"Train size : {len(X_train)}")
    print(f"Test size  : {len(X_test)}")
    print(f"Sample review:\n{X_train.iloc[0]}")
