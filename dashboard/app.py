"""
SentimentScribe — Streamlit business dashboard.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = Path(__file__).parents[1] / "data" / "bookReviewsData.csv"

st.set_page_config(
    page_title="SentimentScribe Dashboard",
    page_icon="📚",
    layout="wide",
)

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, header=0)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["label"] = df["positive_review"].astype(int)
    df["sentiment"] = df["label"].map({1: "Positive", 0: "Negative"})
    df["text_length"] = df["review"].str.split().str.len()
    return df


def get_top_keywords(df: pd.DataFrame, sentiment: int, n: int = 20) -> list[tuple[str, int]]:
    """Return top n keywords for a given sentiment label."""
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "is", "it", "in",
        "of", "to", "for", "this", "that", "was", "with", "as",
        "are", "be", "by", "on", "at", "from", "its", "not", "have",
        "had", "has", "he", "she", "they", "his", "her", "their",
        "book", "read", "author", "story", "one", "just", "very",
        "i", "me", "my", "we", "you", "your", "so", "if", "no",
        "about", "more", "there", "when", "what", "which", "who",
        "than", "do", "did", "been", "also", "even", "would",
        "could", "should", "really", "like", "get", "will", "all",
    }
    subset = df[df["label"] == sentiment]["review"].dropna()
    words = re.findall(r"\b[a-z]{4,}\b", " ".join(subset.str.lower()))
    counter = Counter(w for w in words if w not in STOPWORDS)
    return counter.most_common(n)


def call_api(review: str) -> dict:
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"review": review},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


# ── Layout ─────────────────────────────────────────────────────────────────────
st.title("📚 SentimentScribe — Book Review Intelligence Dashboard")
st.caption(
    "Production ML system for sentiment analysis on book reviews. "
    "Models: Logistic Regression · Naive Bayes · DistilBERT"
)

# Check API status
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    api_ok = health.get("model_loaded", False)
except Exception:
    api_ok = False

col_status1, col_status2 = st.columns([1, 4])
with col_status1:
    if api_ok:
        st.success("API Online", icon="✅")
    else:
        st.warning("API Offline — demo mode", icon="⚠️")

st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────
try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    st.error(
        "bookReviewsData.csv not found in data/. "
        "Place the dataset file there to enable analytics."
    )
    data_loaded = False

if data_loaded:
    # ── KPI row ───────────────────────────────────────────────────────────────
    total = len(df)
    n_pos = int(df["label"].sum())
    n_neg = total - n_pos
    avg_len = int(df["text_length"].mean())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Reviews", f"{total:,}")
    k2.metric("Positive", f"{n_pos:,}", f"{n_pos/total*100:.1f}%")
    k3.metric("Negative", f"{n_neg:,}", f"{-n_neg/total*100:.1f}%")
    k4.metric("Avg Review Length", f"{avg_len} words")

    st.divider()

    # ── Charts row ────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(
            df,
            names="sentiment",
            color="sentiment",
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Review Length Distribution by Sentiment")
        fig_hist = px.histogram(
            df,
            x="text_length",
            color="sentiment",
            nbins=50,
            opacity=0.75,
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
            barmode="overlay",
        )
        fig_hist.update_layout(xaxis_title="Word Count", margin=dict(t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Keywords row ─────────────────────────────────────────────────────────
    kw1, kw2 = st.columns(2)

    with kw1:
        st.subheader("Top Keywords — Negative Reviews")
        neg_kw = get_top_keywords(df, 0, n=15)
        neg_df = pd.DataFrame(neg_kw, columns=["keyword", "count"])
        fig_neg = px.bar(
            neg_df.sort_values("count"),
            x="count", y="keyword", orientation="h",
            color_discrete_sequence=["#e74c3c"],
        )
        fig_neg.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_neg, use_container_width=True)

    with kw2:
        st.subheader("Top Keywords — Positive Reviews")
        pos_kw = get_top_keywords(df, 1, n=15)
        pos_df = pd.DataFrame(pos_kw, columns=["keyword", "count"])
        fig_pos = px.bar(
            pos_df.sort_values("count"),
            x="count", y="keyword", orientation="h",
            color_discrete_sequence=["#2ecc71"],
        )
        fig_pos.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_pos, use_container_width=True)

st.divider()

# ── Live prediction demo ──────────────────────────────────────────────────────
st.subheader("Live Prediction Demo")
st.caption("Enter a book review and get an instant sentiment prediction.")

demo_col1, demo_col2 = st.columns([3, 1])
with demo_col1:
    user_review = st.text_area(
        "Enter a book review:",
        placeholder="e.g. This was an absolutely wonderful read — deeply moving and beautifully written.",
        height=120,
    )

with demo_col2:
    st.write("")
    st.write("")
    predict_btn = st.button("Predict Sentiment", type="primary", use_container_width=True)

if predict_btn and user_review.strip():
    if api_ok:
        result = call_api(user_review)
        if "error" in result:
            st.error(f"API error: {result['error']}")
        else:
            sentiment = result["sentiment"].capitalize()
            conf = result["confidence"]
            color = "#2ecc71" if result["sentiment"] == "positive" else "#e74c3c"
            icon = "😊" if result["sentiment"] == "positive" else "😞"
            st.markdown(
                f"""
                <div style="border-radius:10px; padding:20px; background:{color}22; border:2px solid {color}">
                    <h2 style="color:{color}; margin:0">{icon} {sentiment}</h2>
                    <p style="margin:8px 0 0 0; font-size:1.1em">Confidence: <b>{conf:.1%}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # Offline demo using a simple heuristic
        positive_words = {"good", "great", "excellent", "love", "wonderful", "amazing", "best", "fantastic", "beautiful"}
        negative_words = {"bad", "terrible", "awful", "hate", "boring", "disappointing", "worst", "poor", "waste"}
        words = set(user_review.lower().split())
        pos_score = len(words & positive_words)
        neg_score = len(words & negative_words)
        sentiment = "Positive" if pos_score >= neg_score else "Negative"
        st.info(f"Demo mode (API offline): **{sentiment}** (heuristic only)")
elif predict_btn:
    st.warning("Please enter a review before clicking Predict.")
