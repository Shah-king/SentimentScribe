"""
SentimentScribe — FastAPI inference service.

Endpoints
---------
GET  /health              → liveness probe
GET  /model-info          → current model metadata
POST /predict             → single prediction (optional auth)
POST /batch               → batch predictions (requires API key or JWT)
POST /bulk-analyze        → CSV upload → summary + shareable report_id (requires auth)
GET  /report/{report_id}  → public shareable report (no auth)
POST /api-keys            → generate a new API key (requires JWT)
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from api.auth import CurrentUser, optional_auth, require_api_key
from src.inference.predictor import SentimentPredictor, get_predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Environment ────────────────────────────────────────────────────────────────
MODEL_TYPE: Literal["sklearn", "distilbert"] = os.getenv("MODEL_TYPE", "sklearn")  # type: ignore
ARTIFACTS_DIR: str = os.getenv("ARTIFACTS_DIR", "experiments/artifacts")
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    review: str = Field(..., min_length=5, max_length=10_000)

    @field_validator("review")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class PredictResponse(BaseModel):
    sentiment: Literal["positive", "negative"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_type: str


class BatchPredictRequest(BaseModel):
    reviews: List[str] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    artifacts_dir: str


class BulkAnalyzeResponse(BaseModel):
    report_id: str
    total: int
    positive: int
    negative: int
    positive_pct: float
    negative_pct: float


class ReportResponse(BaseModel):
    report_id: str
    summary: Dict[str, Any]
    created_at: str


class ApiKeyResponse(BaseModel):
    key: str
    name: str
    message: str


# ── App lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model at startup (type=%s)…", MODEL_TYPE)
    try:
        get_predictor(model_type=MODEL_TYPE, artifacts_dir=ARTIFACTS_DIR)
        logger.info("Model loaded successfully.")
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
    yield
    logger.info("Shutting down SentimentScribe API.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SentimentScribe API",
    description="Production-ready book review sentiment analysis service.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_loaded_predictor() -> SentimentPredictor:
    try:
        return get_predictor(model_type=MODEL_TYPE, artifacts_dir=ARTIFACTS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {exc}. Train a model first with train.py",
        )


async def _supabase_insert(table: str, data: dict) -> None:
    """Insert a row into a Supabase table (best-effort, non-blocking)."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, headers=headers, json=data)
    except Exception as exc:
        logger.warning("Supabase insert to %s failed: %s", table, exc)


async def _supabase_get(table: str, params: dict) -> list:
    """Query a Supabase table and return rows."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return []
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    }
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("Supabase get from %s failed: %s", table, exc)
        return []


# ── Ops endpoints ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe."""
    try:
        get_predictor(model_type=MODEL_TYPE, artifacts_dir=ARTIFACTS_DIR)
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(model_type=MODEL_TYPE, artifacts_dir=ARTIFACTS_DIR)


# ── Inference endpoints ────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    request: PredictRequest,
    current_user: Optional[CurrentUser] = Depends(optional_auth),
) -> PredictResponse:
    """Predict sentiment for a single book review.

    Public endpoint — no auth required.
    Providing a valid API key or Bearer token records usage analytics.

    **Example**
    ```json
    {"review": "A fascinating and deeply moving story."}
    ```
    """
    predictor = _get_loaded_predictor()
    sentiment, confidence = predictor.predict(request.review)
    return PredictResponse(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        model_type=MODEL_TYPE,
    )


@app.post("/batch", response_model=BatchPredictResponse, tags=["inference"])
async def batch_predict(
    request: BatchPredictRequest,
    current_user: CurrentUser = Depends(require_api_key),
) -> BatchPredictResponse:
    """Predict sentiment for a batch of reviews (max 100).

    **Requires** an `X-API-Key` header or `Authorization: Bearer <jwt>`.
    Generate a key in the SentimentScribe dashboard → Developer tab.
    """
    predictor = _get_loaded_predictor()
    results: List[PredictResponse] = []
    for review in request.reviews:
        sentiment, confidence = predictor.predict(review)
        results.append(
            PredictResponse(
                sentiment=sentiment,
                confidence=round(confidence, 4),
                model_type=MODEL_TYPE,
            )
        )
    return BatchPredictResponse(predictions=results)


@app.post("/bulk-analyze", response_model=BulkAnalyzeResponse, tags=["inference"])
async def bulk_analyze(
    file: UploadFile = File(..., description="CSV with a 'review' column"),
    current_user: CurrentUser = Depends(require_api_key),
) -> BulkAnalyzeResponse:
    """Upload a CSV file and get back a bulk sentiment summary.

    The CSV must have a column named `review` (case-insensitive).
    Returns a `report_id` that can be shared via `GET /report/{report_id}`.

    **Requires** an `X-API-Key` header or `Authorization: Bearer <jwt>`.
    """
    import pandas as pd

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    df.columns = df.columns.str.strip().str.lower()
    if "review" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a column named 'review'.",
        )

    reviews = df["review"].fillna("").tolist()[:500]
    predictor = _get_loaded_predictor()

    sentiments = [predictor.predict(r)[0] for r in reviews]
    n_pos = sum(1 for s in sentiments if s == "positive")
    n_neg = len(sentiments) - n_pos
    total = len(sentiments)

    report_id = str(uuid.uuid4())
    summary = {
        "total": total,
        "positive": n_pos,
        "negative": n_neg,
        "positive_pct": round(n_pos / total * 100, 1) if total else 0,
        "negative_pct": round(n_neg / total * 100, 1) if total else 0,
        "filename": file.filename,
    }

    await _supabase_insert("reports", {
        "id": report_id,
        "user_id": current_user.user_id,
        "summary": summary,
    })

    return BulkAnalyzeResponse(report_id=report_id, **summary)


@app.get("/report/{report_id}", response_model=ReportResponse, tags=["inference"])
async def get_report(report_id: str) -> ReportResponse:
    """Retrieve a shareable bulk analysis report by ID.

    This endpoint is **public** — no auth required.
    Anyone with the report_id can view the summary.
    """
    rows = await _supabase_get(
        "reports",
        {"id": f"eq.{report_id}", "select": "id,summary,created_at"},
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Report not found.")
    row = rows[0]
    return ReportResponse(
        report_id=row["id"],
        summary=row["summary"],
        created_at=row["created_at"],
    )


# ── API key management ─────────────────────────────────────────────────────────
@app.post("/api-keys", response_model=ApiKeyResponse, tags=["auth"])
async def create_api_key(
    name: str = Form(default="My Key", max_length=50),
    current_user: CurrentUser = Depends(require_api_key),
) -> ApiKeyResponse:
    """Generate a new API key for the authenticated user.

    Requires an existing valid `Authorization: Bearer <jwt>` token.
    The raw key is returned **once** — store it securely.
    """
    raw_key = f"ss_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    await _supabase_insert("api_keys", {
        "user_id": current_user.user_id,
        "key_hash": key_hash,
        "name": name,
    })

    return ApiKeyResponse(
        key=raw_key,
        name=name,
        message="Store this key securely — it will not be shown again.",
    )
