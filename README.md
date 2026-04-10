# SentimentScribe 📚

> **Production-ready Book Review Sentiment Analysis System**
> End-to-end MLOps pipeline · Authenticated REST API · Multi-feature dashboard · Experiment tracking

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-orange.svg)](https://mlflow.org/)
[![Supabase](https://img.shields.io/badge/Supabase-Auth%20%2B%20DB-3ECF8E.svg)](https://supabase.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

SentimentScribe transforms raw book reviews into actionable business intelligence.
The system classifies reviews as **positive** or **negative** using a production ML pipeline with user authentication, persistent history, bulk analysis, and a developer API — built to scale from prototype to deployment.

**Business value:**
- Automate review moderation at scale
- Surface negative trends before they damage brand reputation
- Power recommendation engines with sentiment signals
- Replace manual review labeling (saves ~$0.10–$0.50 per review at scale)

**Models evaluated:**

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression (TF-IDF) | 82.0% | 0.90 |
| Naive Bayes (TF-IDF) | 81.0% | 0.89 |
| Neural Network (TF-IDF) | 82.7% | 0.91 |
| **DistilBERT (fine-tuned)** | **~91%\*** | **~0.96\*** |

*\*Expected after fine-tuning; actual results vary with hardware/epochs.*

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SentimentScribe System v2                          │
├──────────────┬──────────────┬───────────────────┬────────────────────────┤
│  Data Layer  │  Training    │  Serving Layer    │  Observability         │
│              │  Pipeline    │                   │                        │
│  CSV /       │  train.py    │  FastAPI v2.0     │  MLflow Tracking       │
│  HuggingFace │     ↓        │  /predict (pub)   │  Evidently Drift       │
│     ↓        │  TF-IDF      │  /batch  (key)    │  SHAP Explainer        │
│  load_data   │  +sklearn    │  /bulk-analyze    │  Streamlit Dashboard   │
│  clean_and   │  or          │  /report/{id}     │    ├─ History          │
│  _label      │  DistilBERT  │  /api-keys        │    ├─ Book Analyzer    │
│              │              │       ↓           │    ├─ Bulk Upload      │
│              │              │  Supabase Postgres│    ├─ Compare          │
│              │              │  (auth + data)    │    └─ Developer        │
└──────────────┴──────────────┴───────────────────┴────────────────────────┘
```

---

## Project Structure

```
SentimentScribe/
│
├── data/
│   └── bookReviewsData.csv
│
├── src/
│   ├── data_pipeline/
│   │   └── load_data.py           # Reproducible data loader (extracted from notebook)
│   ├── features/
│   │   └── text_preprocessor.py  # TF-IDF feature engineering
│   ├── models/
│   │   ├── tfidf_models.py        # Logistic Regression + Naive Bayes wrappers
│   │   └── transformer_model.py  # DistilBERT fine-tuning (HuggingFace)
│   ├── training/
│   │   └── trainer.py             # MLflow-integrated training pipeline
│   ├── inference/
│   │   └── predictor.py           # Versioned model loader + cached inference
│   ├── monitoring/
│   │   └── drift_detector.py      # Evidently AI drift detection
│   └── utils/
│       ├── logger.py
│       └── explainability.py     # SHAP feature importance
│
├── api/
│   ├── main.py                    # FastAPI service — 7 endpoints
│   └── auth.py                    # JWT + API key verification (Depends)
│
├── dashboard/
│   ├── app.py                     # Streamlit multi-page authenticated dashboard
│   └── auth.py                    # Supabase login / signup / logout
│
├── supabase/
│   └── migrations/
│       └── 001_initial.sql        # predictions, api_keys, reports, bulk_jobs tables
│
├── configs/
│   └── config.yaml                # All hyperparameters, feature flags, Supabase config
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_api.py
│   └── test_model.py
│
├── docker/
│   └── Dockerfile
│
├── .streamlit/
│   └── secrets.toml.example       # Template for local secrets (never commit secrets.toml)
│
├── train.py                       # Training entry point
├── monitor.py                     # Drift monitoring entry point
├── docker-compose.yml
├── render.yaml                    # Render web service config
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Shah-king/SentimentScribe.git
cd SentimentScribe

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set up Supabase (free)

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** → paste and run `supabase/migrations/001_initial.sql`
3. Go to **Settings → API** → copy your **Project URL** and **anon key**

Create `.streamlit/secrets.toml`:

```toml
SUPABASE_URL      = "https://your-project-ref.supabase.co"
SUPABASE_ANON_KEY = "eyJ..."
API_URL           = "https://your-render-service.onrender.com"  # or http://localhost:8000 for local dev
```

### 3. Add the dataset

```
data/
└── bookReviewsData.csv    # columns: Review, Positive Review
```

### 4. Train a model

```bash
# Logistic Regression (fast, ~30 seconds)
python train.py --model logistic_regression

# DistilBERT (requires GPU + torch)
pip install torch transformers datasets
python train.py --model distilbert
```

MLflow UI: `mlflow ui` → `http://localhost:5000`

### 5. Start the API

```bash
uvicorn api.main:app --reload      # → http://localhost:8000/docs
```

### 6. Start the dashboard

```bash
streamlit run dashboard/app.py     # → http://localhost:8501
```

---

## Dashboard Features

After signing in, users have access to six pages:

| Page | Description | Auth required |
|---|---|---|
| **Dashboard** | Sentiment KPIs, distribution charts, keyword analysis, live prediction demo | Login |
| **History** | Personal log of every prediction with trend chart and export | Login |
| **Book Analyzer** | Search the dataset by keyword or title; see aggregated sentiment | Login |
| **Bulk Upload** | Upload a CSV of reviews → enriched download + shareable report link | Login |
| **Compare** | Side-by-side sentiment comparison of two reviews | Login |
| **Developer** | Generate and manage personal API keys | Login |

---

## API Reference

### `POST /predict` — public

Classify a single review. No auth required.

```json
// Request
{ "review": "A deeply moving and beautifully crafted novel." }

// Response
{ "sentiment": "positive", "confidence": 0.9241, "model_type": "sklearn" }
```

### `POST /batch` — requires API key

Classify up to 100 reviews. Requires `X-API-Key` header.

```bash
curl -X POST http://localhost:8000/batch \
  -H "X-API-Key: ss_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great book!", "Terrible waste of time."]}'
```

### `POST /bulk-analyze` — requires API key

Upload a CSV file and receive a sentiment summary + shareable `report_id`.

```bash
curl -X POST http://localhost:8000/bulk-analyze \
  -H "X-API-Key: ss_your_key_here" \
  -F "file=@reviews.csv"
```

```json
{
  "report_id": "uuid",
  "total": 150,
  "positive": 112,
  "negative": 38,
  "positive_pct": 74.7,
  "negative_pct": 25.3
}
```

### `GET /report/{report_id}` — public

Retrieve a shareable bulk analysis report. No auth required — anyone with the ID can view it.

### `POST /api-keys` — requires JWT

Generate a new API key from the dashboard or programmatically.

### `GET /health` · `GET /model-info` — public

Ops probes for liveness and model metadata.

---

## Authentication

SentimentScribe uses **Supabase Auth** for user identity and **Supabase Postgres** for data persistence.

| Layer | Implementation |
|---|---|
| User login / signup | Supabase email + password auth via `supabase-py` |
| Session management | `st.session_state` stores JWT on successful login |
| API key auth | SHA-256 hashed keys stored in `api_keys` table; verified per-request |
| JWT verification | FastAPI `Depends(require_api_key)` decodes Supabase-issued JWTs via `python-jose` |
| Data isolation | Supabase Row Level Security — users only see their own rows |

---

## Deployment

### Render (FastAPI)

The inference API is deployed to Render as a Python web service.

1. Connect your GitHub repo on [dashboard.render.com](https://dashboard.render.com) → **New → Web Service**
2. Set the build command to `pip install -r requirements.txt`
3. Set the start command to `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add these environment variables in the Render dashboard:

| Variable | Where to find it |
|---|---|
| `SUPABASE_URL` | Supabase → Project Settings → API → Project URL |
| `SUPABASE_ANON_KEY` | Supabase → Project Settings → API → anon public key |
| `SUPABASE_JWT_SECRET` | Supabase → Project Settings → API → JWT Secret |

The `render.yaml` at the repo root documents the full service config.

### Streamlit Community Cloud (Dashboard)

The dashboard is deployed via [share.streamlit.io](https://share.streamlit.io). Set the following in **App Settings → Secrets**:

```toml
SUPABASE_URL      = "https://your-project-ref.supabase.co"
SUPABASE_ANON_KEY = "eyJ..."
API_URL           = "https://your-render-service.onrender.com"
```

### Docker (local full stack)

```bash
cp .env.example .env       # fill in SUPABASE_URL, SUPABASE_ANON_KEY
docker compose up --build
```

| Service | URL |
|---|---|
| Inference API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

---

## Monitoring & Drift Detection

```bash
python monitor.py --type data --current data/new_reviews.csv
python monitor.py --type predictions --current data/new_predictions.csv
```

HTML reports written to `reports/`.

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Configuration

All behaviour is controlled via `configs/config.yaml` — no code changes needed to switch models, toggle features, or adjust hyperparameters.

```yaml
features:
  require_auth_for_predict: false   # keep public demo open
  require_auth_for_batch: true      # batch requires API key

models:
  logistic_regression:
    max_iter: 1000
    C: 1.0
  distilbert:
    num_epochs: 3
    batch_size: 16
    learning_rate: 2.0e-5
```

---

## Key Engineering Decisions

| Decision | Rationale |
|---|---|
| Notebook → modular package | Enables CI/CD, testing, and reproducibility |
| Supabase over SQLite/JWT-DIY | Free Postgres + Auth + RLS; works on Streamlit Cloud without a persistent filesystem |
| Row Level Security on all tables | Users can never read each other's data even with the public anon key exposed |
| `optional_auth` on `/predict` | Keeps the public demo accessible while enabling usage tracking for authenticated users |
| `@lru_cache` on predictor | Single model load per process; thread-safe |
| SHA-256 API key hashing | Raw key shown once at generation; only the hash is stored |
| Config feature flags | `require_auth_for_predict` can flip without a redeploy |
| Pydantic v2 validation | Runtime type safety on all API inputs |

---

## Roadmap

- [ ] ONNX export for faster CPU inference
- [ ] GitHub Actions CI/CD with automated test gates
- [ ] Email alerts when drift is detected
- [ ] A/B testing framework for model comparison
- [ ] Kubernetes Helm chart

---

## License

MIT © 2025 Shah-king
