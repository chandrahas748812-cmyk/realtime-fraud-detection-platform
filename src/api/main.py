"""Real-time fraud scoring API.

Exposes POST /score: validates an incoming transaction with Pydantic,
builds model features, runs the (pluggable) fraud model and returns a
fraud probability, a risk tier and the inference latency in milliseconds.

Run locally with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import time
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# TODO: replace the stub scorer below with the trained XGBoost artefact
# produced by `src/models/train.py` (e.g. load `artifacts/model.json`
# at startup and call `model.predict_proba` here).
MODEL_VERSION = "0.1.0-stub"


class Transaction(BaseModel):
    """Incoming transaction payload."""

    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant: str = Field(..., min_length=1, description="Merchant name")
    merchant_category: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    card_last4: str = Field(..., min_length=4, max_length=4)
    timestamp: str = Field(..., description="ISO-8601 event time")
    # Optional behavioural context looked up from the feature store.
    user_avg_amount_30d: Optional[float] = Field(default=None, gt=0)
    user_txn_count_24h: Optional[int] = Field(default=None, ge=0)
    last_latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    last_longitude: Optional[float] = Field(default=None, ge=-180, le=180)


class ScoreResponse(BaseModel):
    """Fraud score returned for a transaction."""

    fraud_probability: float = Field(..., ge=0, le=1)
    risk_tier: Literal["low", "medium", "high"]
    latency_ms: float
    model_version: str


def risk_tier(probability: float) -> str:
    """Map a fraud probability to a human-readable risk tier."""
    if probability >= 0.7:
        return "high"
    if probability >= 0.2:
        return "medium"
    return "low"


def stub_score(txn: Transaction) -> float:
    """Placeholder scorer used until a trained model is wired in.

    Combines a few simple heuristics (large amount, high 24h velocity)
    so the API contract can be exercised end to end.

    TODO: delete this function and call the trained XGBoost model instead.
    """
    score = 0.02  # base rate prior
    if txn.amount > 1000:
        score += 0.25
    if txn.user_txn_count_24h and txn.user_txn_count_24h > 10:
        score += 0.30
    if txn.user_avg_amount_30d and txn.amount > 3 * txn.user_avg_amount_30d:
        score += 0.25
    return float(min(max(score, 0.0), 0.99))


app = FastAPI(
    title="Realtime Fraud Detection API",
    description="Scores banking transactions for fraud in real time.",
    version=MODEL_VERSION,
)


@app.get("/health")
def health() -> dict:
    """Liveness probe for load balancers / orchestrators."""
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    """Score a single transaction for fraud.

    Returns the fraud probability, a risk tier and the inference
    latency in milliseconds. Target: sub-100ms p99.
    """
    start = time.perf_counter()
    try:
        probability = stub_score(txn)
    except Exception as exc:  # never leak a 500 on a scoring failure
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - start) * 1000
    return ScoreResponse(
        fraud_probability=round(probability, 4),
        risk_tier=risk_tier(probability),
        latency_ms=round(latency_ms, 2),
        model_version=MODEL_VERSION,
    )
