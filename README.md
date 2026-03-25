# SentimentScribe 📚

> **Production-ready Book Review Sentiment Analysis System**
> End-to-end MLOps pipeline · REST API · Real-time dashboard · Experiment tracking

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-orange.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

SentimentScribe transforms raw book reviews into actionable business intelligence.
The system classifies reviews as **positive** or **negative** using a production ML pipeline that scales from prototype to deployment.

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
┌─────────────────────────────────────────────────────────────────┐
│                     SentimentScribe System                       │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Data Layer  │ Training     │  Serving     │  Observability     │
│              │  Pipeline    │  Layer       │                    │
│  CSV /       │  train.py    │  FastAPI     │  MLflow Tracking   │
│  HuggingFace │     ↓        │  /predict    │  Evidently Drift   │
│     ↓        │  TF-IDF      │  /batch      │  SHAP Explainer    │
│  load_data   │  +sklearn    │     ↓        │  Streamlit Dash    │
│  clean_and   │  or          │  Predictor   │                    │
│  _label      │  DistilBERT  │  (cached)    │                    │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

---

## Project Structure

```
SentimentScribe/
│
├── data/                          # Dataset files
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
│   │   └── predictor.py           # Versioned model loader + inference
│   ├── monitoring/
│   │   └── drift_detector.py      # Evidently AI drift detection
│   └── utils/
│       ├── logger.py              # Centralised logging
│       └── explainability.py     # SHAP feature importance
│
├── api/
│   └── main.py                    # FastAPI service (POST /predict, POST /batch)
│
├── dashboard/
│   └── app.py                     # Streamlit business dashboard
│
├── configs/
│   └── config.yaml                # All hyperparameters & settings
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_api.py
│   └── test_model.py
│
├── docker/
│   └── Dockerfile
│
├── train.py                       # Training entry point
├── monitor.py                     # Drift monitoring entry point
├── docker-compose.yml             # Full stack orchestration
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

### 2. Add the dataset

Place `bookReviewsData.csv` inside `data/`:

```
data/
└── bookReviewsData.csv    # columns: Review, Positive Review
```

### 3. Train a model

```bash
# Logistic Regression (fast, recommended first run)
python train.py --model logistic_regression

# Naive Bayes
python train.py --model naive_bayes

# DistilBERT (requires GPU, torch, transformers)
pip install torch transformers datasets
python train.py --model distilbert
```

MLflow logs are written to `mlruns/`. Launch the UI:

```bash
mlflow ui                          # → http://localhost:5000
```

### 4. Start the API

```bash
uvicorn api.main:app --reload      # → http://localhost:8000
```

Interactive docs: `http://localhost:8000/docs`

### 5. Start the dashboard

```bash
pip install streamlit plotly
streamlit run dashboard/app.py     # → http://localhost:8501
```

---

## API Reference

### `POST /predict`

Classify a single review.

**Request**
```json
{
  "review": "A deeply moving and beautifully crafted novel."
}
```

**Response**
```json
{
  "sentiment": "positive",
  "confidence": 0.9241,
  "model_type": "sklearn"
}
```

### `POST /batch`

Classify up to 100 reviews in one request.

**Request**
```json
{
  "reviews": [
    "Wonderful story, highly recommended.",
    "Boring and poorly written."
  ]
}
```

**Response**
```json
{
  "predictions": [
    {"sentiment": "positive", "confidence": 0.91, "model_type": "sklearn"},
    {"sentiment": "negative", "confidence": 0.87, "model_type": "sklearn"}
  ]
}
```

### `GET /health`

Liveness probe — returns model load status.

### `GET /model-info`

Returns current model type and artifact directory.

---

## Docker Deployment

Run the full stack (API + Dashboard + MLflow) with a single command:

```bash
# Copy env file
cp .env.example .env

# Build and start all services
docker compose up --build
```

| Service | URL |
|---|---|
| Inference API | http://localhost:8000 |
| Streamlit Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## Monitoring & Drift Detection

After collecting production traffic, detect feature or prediction drift:

```bash
# Data drift (new reviews vs training distribution)
python monitor.py --type data --current data/new_reviews.csv

# Prediction drift (new predictions vs baseline)
python monitor.py --type predictions --current data/new_predictions.csv
```

HTML reports are written to `reports/`.

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Configuration

All hyperparameters are controlled via `configs/config.yaml`.
No code changes needed to switch models, adjust feature settings, or change split ratios.

```yaml
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
| Extracted notebook → modules | Enables CI/CD, testing, and reproducibility |
| Config-driven training | No code changes to tune hyperparameters |
| `@lru_cache` on predictor | Single model load per process; thread-safe |
| Pydantic v2 validation | Runtime type safety on all API inputs |
| Stratified train/test split | Preserves class balance in both splits |
| HuggingFace DistilBERT | 40% smaller than BERT-base, 97% of performance |

---

## Roadmap

- [ ] Add ONNX export for faster CPU inference
- [ ] Redis caching layer for repeated predictions
- [ ] GitHub Actions CI/CD pipeline
- [ ] Kubernetes Helm chart
- [ ] A/B testing framework for model comparison

---

## License

MIT © 2025 Shah-king
