"""
SentimentScribe — Streamlit dashboard (authenticated, multi-feature).

Pages (sidebar nav):
  Dashboard     — analytics overview (public after login)
  History       — user's personal prediction log
  Book Analyzer — search reviews by keyword / title
  Bulk Upload   — CSV upload → enriched download + shareable report
  Compare       — side-by-side review comparison
  Developer     — API key management

Run locally:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
import os
import re
import secrets
import sys
import uuid
from collections import Counter
from pathlib import Path

# Make project root importable regardless of where Streamlit is launched from
_PROJECT_ROOT = Path(__file__).parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from dashboard.auth import get_current_user, get_supabase, logout, require_auth

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

DATA_PATH = Path(__file__).parents[1] / "data" / "bookReviewsData.csv"
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).parent / "data" / "bookReviewsData.csv"

st.set_page_config(
    page_title="SentimentScribe",
    page_icon="📚",
    layout="wide",
)

# ── Auth gate — must be before any other rendering ────────────────────────────
require_auth()
user = get_current_user()

# ── Shared helpers ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, header=0)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["label"] = df["positive_review"].astype(int)
    df["sentiment"] = df["label"].map({1: "Positive", 0: "Negative"})
    df["text_length"] = df["review"].str.split().str.len()
    return df


def get_top_keywords(df: pd.DataFrame, sentiment: int, n: int = 15) -> list[tuple[str, int]]:
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "is", "it", "in", "of", "to",
        "for", "this", "that", "was", "with", "as", "are", "be", "by", "on",
        "at", "from", "its", "not", "have", "had", "has", "he", "she", "they",
        "his", "her", "their", "book", "read", "author", "story", "one", "just",
        "very", "i", "me", "my", "we", "you", "your", "so", "if", "no", "about",
        "more", "there", "when", "what", "which", "who", "than", "do", "did",
        "been", "also", "even", "would", "could", "should", "really", "like",
        "get", "will", "all", "some", "time", "into", "than", "after", "first",
    }
    subset = df[df["label"] == sentiment]["review"].dropna()
    words = re.findall(r"\b[a-z]{4,}\b", " ".join(subset.str.lower()))
    counter = Counter(w for w in words if w not in STOPWORDS)
    return counter.most_common(n)


def call_api(review: str) -> dict:
    token = st.session_state.get("access_token")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"review": review},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


def _save_prediction(review: str, sentiment: str, confidence: float, model_type: str) -> None:
    """Persist a prediction to Supabase (best-effort, never blocks the UI)."""
    try:
        supabase = get_supabase()
        supabase.table("predictions").insert({
            "user_id": user.id,
            "review_text": review[:1000],
            "sentiment": sentiment,
            "confidence": confidence,
            "model_type": model_type,
        }).execute()
    except Exception:
        pass


def _render_result_card(sentiment: str, confidence: float, subtitle: str = "") -> None:
    color = "#2ecc71" if sentiment.lower() == "positive" else "#e74c3c"
    icon = "😊" if sentiment.lower() == "positive" else "😞"
    sub_html = f'<p style="margin:6px 0 0 0; color:#888; font-size:0.95em">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div style="border-radius:10px; padding:20px; background:{color}22; border:2px solid {color}">
            <h2 style="color:{color}; margin:0">{icon} {sentiment.capitalize()}</h2>
            <p style="margin:8px 0 0 0; font-size:1.1em">Confidence: <b>{confidence:.1%}</b></p>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _keyword_fallback(review: str) -> tuple[str, float]:
    """Offline heuristic when API is unreachable."""
    pos = {"good", "great", "excellent", "love", "wonderful", "amazing", "best", "fantastic", "beautiful"}
    neg = {"bad", "terrible", "awful", "hate", "boring", "disappointing", "worst", "poor", "waste"}
    words = set(review.lower().split())
    sentiment = "positive" if len(words & pos) >= len(words & neg) else "negative"
    return sentiment, 0.0


# ── Check API status ──────────────────────────────────────────────────────────
try:
    _health = requests.get(f"{API_URL}/health", timeout=3).json()
    api_ok = _health.get("model_loaded", False)
except Exception:
    api_ok = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 SentimentScribe")
    st.caption(f"Signed in as **{user.email}**")
    if st.button("Sign Out", use_container_width=True):
        logout()

    st.divider()
    page = st.radio(
        "Navigate",
        ["Dashboard", "History", "Book Analyzer", "Bulk Upload", "Compare", "Developer"],
        label_visibility="collapsed",
    )
    st.divider()
    if api_ok:
        st.success("API Online", icon="✅")
    else:
        st.info("API not connected\nPredictions use keyword fallback", icon="ℹ️")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("📚 SentimentScribe")
    st.caption("Book review sentiment analysis · ML-powered insights")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("bookReviewsData.csv not found in data/.")
        st.stop()

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

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(
            df, names="sentiment", color="sentiment",
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
            hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Review Length by Sentiment")
        fig_hist = px.histogram(
            df, x="text_length", color="sentiment", nbins=50, opacity=0.75,
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
            barmode="overlay",
        )
        fig_hist.update_layout(xaxis_title="Word Count", margin=dict(t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    kw1, kw2 = st.columns(2)
    with kw1:
        st.subheader("Top Keywords — Negative")
        neg_df = pd.DataFrame(get_top_keywords(df, 0), columns=["keyword", "count"])
        st.plotly_chart(
            px.bar(neg_df.sort_values("count"), x="count", y="keyword",
                   orientation="h", color_discrete_sequence=["#e74c3c"]),
            use_container_width=True,
        )
    with kw2:
        st.subheader("Top Keywords — Positive")
        pos_df = pd.DataFrame(get_top_keywords(df, 1), columns=["keyword", "count"])
        st.plotly_chart(
            px.bar(pos_df.sort_values("count"), x="count", y="keyword",
                   orientation="h", color_discrete_sequence=["#2ecc71"]),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Live Prediction Demo")
    d1, d2 = st.columns([3, 1])
    with d1:
        demo_review = st.text_area(
            "Enter a book review:",
            placeholder="e.g. A deeply moving story that I could not put down.",
            height=100, key="demo_review",
        )
    with d2:
        st.write("")
        st.write("")
        demo_btn = st.button("Predict", type="primary", use_container_width=True)

    if demo_btn and demo_review.strip():
        if api_ok:
            result = call_api(demo_review)
            if "error" not in result:
                _render_result_card(result["sentiment"], result["confidence"])
                _save_prediction(demo_review, result["sentiment"],
                                 result["confidence"], result.get("model_type", "sklearn"))
            else:
                st.error(f"API error: {result['error']}")
        else:
            sentiment, _ = _keyword_fallback(demo_review)
            _render_result_card(sentiment, 0.0,
                                subtitle="Keyword preview · Deploy API for ML predictions")
    elif demo_btn:
        st.warning("Please enter a review.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: History
# ══════════════════════════════════════════════════════════════════════════════
elif page == "History":
    st.title("My Prediction History")
    st.caption("Every review you've analyzed, saved to your account.")

    try:
        supabase = get_supabase()
        resp = (
            supabase.table("predictions")
            .select("*")
            .eq("user_id", str(user.id))
            .order("created_at", desc=True)
            .limit(200)
            .execute()
        )
        rows = resp.data
    except Exception as exc:
        st.error(f"Could not load history: {exc}")
        st.stop()

    if not rows:
        st.info("No predictions yet. Head to **Dashboard** and analyze a review!")
    else:
        hist_df = pd.DataFrame(rows)
        hist_df["created_at"] = pd.to_datetime(hist_df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        hist_df["confidence"] = hist_df["confidence"].map("{:.1%}".format)

        # KPIs
        total_preds = len(hist_df)
        n_pos_hist = (hist_df["sentiment"] == "positive").sum()
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Analyzed", total_preds)
        h2.metric("Positive", n_pos_hist)
        h3.metric("Negative", total_preds - n_pos_hist)

        st.divider()

        # Trend chart
        trend_df = pd.DataFrame(rows)
        trend_df["date"] = pd.to_datetime(trend_df["created_at"]).dt.date
        trend = trend_df.groupby(["date", "sentiment"]).size().reset_index(name="count")
        if len(trend) > 1:
            st.subheader("Predictions Over Time")
            st.plotly_chart(
                px.line(trend, x="date", y="count", color="sentiment",
                        color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"}),
                use_container_width=True,
            )

        # Table
        st.subheader("All Predictions")
        display_cols = ["created_at", "sentiment", "confidence", "review_text", "model_type"]
        available = [c for c in display_cols if c in hist_df.columns]
        st.dataframe(
            hist_df[available].rename(columns={
                "created_at": "Date", "sentiment": "Sentiment",
                "confidence": "Confidence", "review_text": "Review",
                "model_type": "Model",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Book Analyzer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Book Analyzer":
    st.title("Book Analyzer")
    st.caption("Search the dataset by keyword or title fragment to see aggregated sentiment.")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("bookReviewsData.csv not found.")
        st.stop()

    keyword = st.text_input(
        "Enter a book title or keyword:",
        placeholder="e.g. Harry Potter, Stephen King, romance, mystery…",
    )

    if keyword.strip():
        mask = df["review"].str.contains(keyword, case=False, na=False)
        filtered = df[mask]

        if filtered.empty:
            st.warning(f"No reviews found containing **{keyword}**. Try a different term.")
        else:
            st.success(f"Found **{len(filtered)}** reviews mentioning \"{keyword}\"")

            a1, a2, a3 = st.columns(3)
            n_pos_f = int(filtered["label"].sum())
            n_neg_f = len(filtered) - n_pos_f
            a1.metric("Reviews Found", len(filtered))
            a2.metric("Positive", n_pos_f, f"{n_pos_f/len(filtered)*100:.0f}%")
            a3.metric("Negative", n_neg_f, f"{-n_neg_f/len(filtered)*100:.0f}%")

            b1, b2 = st.columns(2)
            with b1:
                st.subheader("Sentiment Split")
                st.plotly_chart(
                    px.pie(filtered, names="sentiment", hole=0.4,
                           color="sentiment",
                           color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"}),
                    use_container_width=True,
                )
            with b2:
                st.subheader("Review Length Distribution")
                st.plotly_chart(
                    px.histogram(filtered, x="text_length", color="sentiment", nbins=30,
                                 opacity=0.75, barmode="overlay",
                                 color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"}),
                    use_container_width=True,
                )

            # Sample reviews
            st.subheader("Sample Reviews")
            sample = filtered.sample(min(5, len(filtered)), random_state=42)
            for _, row in sample.iterrows():
                color = "#2ecc71" if row["label"] == 1 else "#e74c3c"
                st.markdown(
                    f'<div style="border-left:4px solid {color}; padding:8px 14px; margin-bottom:8px; background:{color}11; border-radius:4px">'
                    f'<b style="color:{color}">{row["sentiment"]}</b><br>'
                    f'<span style="font-size:0.9em">{row["review"][:300]}{"…" if len(row["review"]) > 300 else ""}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Enter a keyword above to explore the dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Bulk Upload
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Bulk Upload":
    st.title("Bulk Sentiment Analysis")
    st.caption("Upload any CSV of book reviews → download enriched results with sentiment labels.")

    # ── Instructions + example download ──────────────────────────────────────
    with st.expander("How it works + download example CSV", expanded=False):
        st.markdown("""
        **What this does:**
        Upload a CSV file containing book reviews. The ML model will classify each review
        as **positive** or **negative** and return a confidence score.
        You can then download the enriched CSV with the results added.

        **Accepted column names for your review text:**
        `review`, `Review`, `text`, `Text`, `comment`, `Comment`, `description`

        Any other columns in your file (title, author, date, etc.) are preserved as-is.

        **Limits:** 500 rows per upload · CSV files only
        """)
        example_csv = "review,title,author\n"
        example_csv += '"This book completely changed my perspective on life. Beautifully written.",The Alchemist,Paulo Coelho\n'
        example_csv += '"Boring and repetitive. Could not finish it. Total waste of time.",Book 2,Author 2\n'
        example_csv += '"A masterpiece. Every page kept me engaged and the ending was perfect.",Book 3,Author 3\n'
        st.download_button(
            label="Download example CSV",
            data=example_csv,
            file_name="example_reviews.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        try:
            upload_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()

        # Flexible column detection — accept common review column names
        upload_df.columns = upload_df.columns.str.strip()
        col_map = {c.lower(): c for c in upload_df.columns}
        review_col = next(
            (col_map[k] for k in ["review", "text", "comment", "description"] if k in col_map),
            None,
        )

        if review_col is None:
            st.error(
                f"Could not find a review column. Your columns: **{', '.join(upload_df.columns)}**\n\n"
                "Rename your review column to `review`, `text`, or `comment` and re-upload."
            )
            st.stop()

        # Standardise internally
        upload_df = upload_df.rename(columns={review_col: "review"})
        upload_df.columns = upload_df.columns.str.lower()

        if len(upload_df) > 500:
            st.warning("File has more than 500 rows — processing first 500 only.")
            upload_df = upload_df.head(500)

        st.success(f"Loaded **{len(upload_df)}** reviews from `{uploaded.name}`")

        if not api_ok:
            st.warning(
                "The ML API is not connected — results will use a keyword-based fallback "
                "which is less accurate. For full ML predictions, deploy the FastAPI service "
                "and set `API_URL` in your Streamlit secrets.",
                icon="⚠️",
            )

        run_btn = st.button("Run Analysis", type="primary")

        if run_btn:
            reviews = upload_df["review"].fillna("").tolist()
            sentiments, confidences, model_types = [], [], []

            progress = st.progress(0, text="Analyzing reviews…")

            if api_ok:
                # Call /batch in chunks of 100
                chunk_size = 100
                for i in range(0, len(reviews), chunk_size):
                    chunk = reviews[i: i + chunk_size]
                    try:
                        resp = requests.post(
                            f"{API_URL}/batch",
                            json={"reviews": chunk},
                            headers={"Authorization": f"Bearer {st.session_state.get('access_token', '')}"},
                            timeout=30,
                        )
                        resp.raise_for_status()
                        for pred in resp.json()["predictions"]:
                            sentiments.append(pred["sentiment"])
                            confidences.append(pred["confidence"])
                            model_types.append(pred.get("model_type", "sklearn"))
                    except Exception:
                        # fallback for this chunk
                        for r in chunk:
                            s, _ = _keyword_fallback(r)
                            sentiments.append(s)
                            confidences.append(0.0)
                            model_types.append("keyword")
                    progress.progress(min((i + chunk_size) / len(reviews), 1.0))
            else:
                # Full keyword fallback
                for idx, r in enumerate(reviews):
                    s, _ = _keyword_fallback(r)
                    sentiments.append(s)
                    confidences.append(0.0)
                    model_types.append("keyword")
                    progress.progress((idx + 1) / len(reviews))

            progress.empty()

            upload_df["sentiment"] = sentiments
            upload_df["confidence"] = confidences
            upload_df["model_type"] = model_types

            # Summary
            n_pos_b = (upload_df["sentiment"] == "positive").sum()
            n_neg_b = len(upload_df) - n_pos_b
            b1, b2, b3 = st.columns(3)
            b1.metric("Total Analyzed", len(upload_df))
            b2.metric("Positive", n_pos_b)
            b3.metric("Negative", n_neg_b)

            st.plotly_chart(
                px.pie(upload_df, names="sentiment", hole=0.4,
                       color="sentiment",
                       color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"}),
                use_container_width=True,
            )

            # Download button
            csv_bytes = upload_df.to_csv(index=False).encode()
            st.download_button(
                label="Download Enriched CSV",
                data=csv_bytes,
                file_name=f"sentiment_results_{uploaded.name}",
                mime="text/csv",
                type="primary",
            )

            # Save report to Supabase for shareable link
            try:
                report_id = str(uuid.uuid4())
                summary = {
                    "total": len(upload_df),
                    "positive": int(n_pos_b),
                    "negative": int(n_neg_b),
                    "filename": uploaded.name,
                }
                supabase = get_supabase()
                supabase.table("reports").insert({
                    "id": report_id,
                    "user_id": str(user.id),
                    "summary": summary,
                }).execute()
                share_url = f"?report_id={report_id}"
                st.success(f"Report saved! Share link: `{share_url}`")
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Compare
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Compare":
    st.title("Side-by-Side Comparison")
    st.caption("Analyze two reviews simultaneously and compare the results.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Review A")
        review_a = st.text_area("", placeholder="Paste the first review here…",
                                 height=150, key="compare_a", label_visibility="collapsed")
    with col_b:
        st.subheader("Review B")
        review_b = st.text_area("", placeholder="Paste the second review here…",
                                 height=150, key="compare_b", label_visibility="collapsed")

    compare_btn = st.button("Compare", type="primary", use_container_width=True)

    if compare_btn:
        if not review_a.strip() or not review_b.strip():
            st.warning("Please enter both reviews before comparing.")
        else:
            res_a, res_b = {}, {}
            if api_ok:
                res_a = call_api(review_a)
                res_b = call_api(review_b)

            sent_a = res_a.get("sentiment") or _keyword_fallback(review_a)[0]
            conf_a = res_a.get("confidence", 0.0)
            sent_b = res_b.get("sentiment") or _keyword_fallback(review_b)[0]
            conf_b = res_b.get("confidence", 0.0)

            out_a, out_b = st.columns(2)
            with out_a:
                _render_result_card(sent_a, conf_a)
                if not api_ok:
                    st.caption("Keyword preview")
            with out_b:
                _render_result_card(sent_b, conf_b)
                if not api_ok:
                    st.caption("Keyword preview")

            # Save both to history
            if api_ok:
                _save_prediction(review_a, sent_a, conf_a, res_a.get("model_type", "sklearn"))
                _save_prediction(review_b, sent_b, conf_b, res_b.get("model_type", "sklearn"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Developer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Developer":
    st.title("Developer — API Keys")
    st.caption("Generate personal API keys to call the SentimentScribe API from your own code.")

    # Show existing keys
    try:
        supabase = get_supabase()
        keys_resp = (
            supabase.table("api_keys")
            .select("id, name, created_at, last_used_at")
            .eq("user_id", str(user.id))
            .order("created_at", desc=True)
            .execute()
        )
        existing_keys = keys_resp.data
    except Exception as exc:
        st.error(f"Could not load API keys: {exc}")
        existing_keys = []

    if existing_keys:
        st.subheader("Your Keys")
        keys_df = pd.DataFrame(existing_keys)
        keys_df["created_at"] = pd.to_datetime(keys_df["created_at"]).dt.strftime("%Y-%m-%d")
        keys_df["last_used_at"] = pd.to_datetime(keys_df["last_used_at"]).dt.strftime("%Y-%m-%d").fillna("Never")
        st.dataframe(
            keys_df[["name", "created_at", "last_used_at"]].rename(columns={
                "name": "Name", "created_at": "Created", "last_used_at": "Last Used"
            }),
            use_container_width=True, hide_index=True,
        )

        # Revoke
        revoke_id = st.selectbox(
            "Revoke a key",
            options=["— select —"] + [f"{k['name']} ({k['id'][:8]}…)" for k in existing_keys],
        )
        if revoke_id != "— select —" and st.button("Revoke", type="secondary"):
            key_id = existing_keys[[f"{k['name']} ({k['id'][:8]}…)" for k in existing_keys].index(revoke_id)]["id"]
            try:
                supabase.table("api_keys").delete().eq("id", key_id).execute()
                st.success("Key revoked.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not revoke key: {exc}")
    else:
        st.info("You have no API keys yet. Generate one below.")

    st.divider()
    st.subheader("Generate New Key")

    with st.form("new_key_form"):
        key_name = st.text_input("Key name", value="My Key", max_chars=50)
        gen_btn = st.form_submit_button("Generate Key", type="primary")

    if gen_btn:
        raw_key = f"ss_{secrets.token_urlsafe(32)}"
        import hashlib
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        try:
            supabase = get_supabase()
            supabase.table("api_keys").insert({
                "user_id": str(user.id),
                "key_hash": key_hash,
                "name": key_name or "My Key",
            }).execute()
            st.success("Key generated — copy it now, it will not be shown again.")
            st.code(raw_key, language=None)
        except Exception as exc:
            st.error(f"Could not save key: {exc}")

    st.divider()
    st.subheader("Usage Example")
    st.code(
        f"""import requests

response = requests.post(
    "{API_URL}/predict",
    headers={{"X-API-Key": "ss_your_key_here"}},
    json={{"review": "This book was absolutely wonderful."}}
)
print(response.json())
# {{"sentiment": "positive", "confidence": 0.92, "model_type": "sklearn"}}
""",
        language="python",
    )
