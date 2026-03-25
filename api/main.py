"""
SentimentScribe — FastAPI inference service.

Endpoints
---------
POST /predict   → single review prediction
POST /batch     → batch predictions
GET  /health    → liveness probe
GET  /model-info → current model metadata
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.inference.predictor import SentimentPredictor, get_predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration from environment ─────────────────────────────────────────
MODEL_TYPE: Literal["sklearn", "distilbert"] = os.getenv("MODEL_TYPE", "sklearn")  # type: ignore
ARTIFACTS_DIR: str = os.getenv("ARTIFACTS_DIR", "experiments/artifacts")


# ── Pydantic schemas ────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    review: str = Field(..., min_length=5, max_length=10_000, description="Book review text")

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


# ── App lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the model on startup."""
    logger.info("Loading model at startup (type=%s)…", MODEL_TYPE)
    try:
        get_predictor(model_type=MODEL_TYPE, artifacts_dir=ARTIFACTS_DIR)
        logger.info("Model loaded successfully.")
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
    yield
    logger.info("Shutting down SentimentScribe API.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SentimentScribe API",
    description="Production-ready book review sentiment analysis service.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_loaded_predictor() -> SentimentPredictor:
    try:
        return get_predictor(model_type=MODEL_TYPE, artifacts_dir=ARTIFACTS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {exc}. Train a model first with train.py",
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────
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


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict sentiment for a single book review.

    **Example request**
    ```json
    {"review": "A fascinating and deeply moving story."}
    ```

    **Example response**
    ```json
    {"sentiment": "positive", "confidence": 0.92, "model_type": "sklearn"}
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
async def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict sentiment for a batch of reviews (max 100)."""
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
