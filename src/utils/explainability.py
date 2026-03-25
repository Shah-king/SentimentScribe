"""
SHAP-based model explainability.
Works with TF-IDF + sklearn pipelines and saves feature importance plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _check_shap() -> None:
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "shap is required for explainability. Install: pip install shap"
        ) from exc


def explain_predictions(
    model,
    vectorizer,
    texts: list[str],
    max_display: int = 20,
    output_dir: str | Path = "reports/shap",
    sample_index: Optional[int] = None,
) -> None:
    """Generate SHAP explanation plots for TF-IDF + sklearn models.

    Parameters
    ----------
    model:
        Fitted sklearn classifier (must have ``predict_proba``).
    vectorizer:
        Fitted ``TextPreprocessor`` or ``TfidfVectorizer``.
    texts:
        List of review strings to explain.
    max_display:
        Number of top features to show in summary plot.
    output_dir:
        Directory to save PNG plots.
    sample_index:
        If set, also generate a force plot for this single sample index.
    """
    _check_shap()
    import shap
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Vectorizing %d texts for SHAP analysis…", len(texts))
    if hasattr(vectorizer, "transform"):
        X = vectorizer.transform(texts)
    else:
        X = vectorizer.vectorizer.transform(texts)

    X_dense = X.toarray()

    logger.info("Computing SHAP values (LinearExplainer)…")
    explainer = shap.LinearExplainer(model, X_dense, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_dense)

    if hasattr(vectorizer, "get_feature_names"):
        feature_names = vectorizer.get_feature_names()
    else:
        feature_names = vectorizer.get_feature_names_out().tolist()

    # ── Summary bar plot ──────────────────────────────────────────────────
    logger.info("Generating SHAP summary plot…")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_dense,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="bar",
    )
    plt.tight_layout()
    bar_path = output_dir / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP bar plot saved → %s", bar_path)

    # ── Beeswarm plot ─────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_dense,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    bee_path = output_dir / "shap_summary_beeswarm.png"
    plt.savefig(bee_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP beeswarm plot saved → %s", bee_path)

    # ── Single-sample force plot ──────────────────────────────────────────
    if sample_index is not None:
        logger.info("Generating SHAP force plot for sample %d…", sample_index)
        shap.initjs()
        force = shap.force_plot(
            explainer.expected_value,
            shap_values[sample_index],
            X_dense[sample_index],
            feature_names=feature_names,
            show=False,
            matplotlib=True,
        )
        force_path = output_dir / f"shap_force_sample_{sample_index}.png"
        plt.savefig(force_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP force plot saved → %s", force_path)
