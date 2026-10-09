# Real-Time Fraud Detection Platform

A portfolio-quality, end-to-end real-time fraud detection system for banking
transactions. It ingests payment events as they happen, enriches them with
behavioural and contextual features, scores each transaction with an XGBoost
classifier, and returns a fraud probability, risk tier and explanation in a
sub-100ms p99 inference API.

## Architecture

```text
+--------+     +-------+     +-----------+     +--------+
| Sources|---->| Kafka |---->| Features  |---->| Redis  |
| (core, |     |(topics)|     |(velocity, |     |(store) |
| rails) |     +-------+     | geo, z)   |     +---+----+
+--------+                   +-----+-----+         |   |
                                     |             |   |
                                     v             v   v
                              +----------+   +------------+
                              | Training |   | FastAPI    |
                              | snapshots|   | /score     |
                              +----+-----+   | (XGBoost,  |
                                   |         |  SHAP)     |
                                   v         +-----+------+
                              +--------------------------+
                              | Drift monitoring (PSI) + |
                              | eval harness           |
                              +--------------------------+
```

## Features

- **Streaming ingestion** -- transaction events are published to Kafka
  topics and consumed in near real time.
- **Feature store** -- Redis-backed online store holding per-user and
  per-merchant state (recent amounts, locations, counts) for fast lookup
  at scoring time.
- **Feature engineering** -- velocity features, amount z-score vs user
  history, geo-distance from the last transaction, time-of-day risk and
  merchant risk scores (`src/features/engineering.py`).
- **XGBoost model** -- gradient-boosted classifier trained with class
  weighting for the imbalanced fraud problem; experiments and artefacts
  tracked in MLflow (`src/models/train.py`).
- **FastAPI inference API** -- `POST /score` validates a transaction
  payload with Pydantic and returns a fraud probability, risk tier and
  inference latency. Target: sub-100ms p99 (`src/api/main.py`).
- **SHAP explainability** -- per-transaction feature attributions for
  analyst review and dispute handling.
- **Drift monitoring** -- Population Stability Index (PSI) checks compare
  recent feature distributions against the training baseline and flag
  drift (`src/monitoring/drift.py`).
- **Eval harness** -- precision/recall at top-k alerts, PR-AUC and latency
  benchmarks to validate every model iteration.

## Tech Stack

| Layer               | Technology            |
|---------------------|-----------------------|
| Language            | Python 3.11           |
| API                 | FastAPI + Uvicorn     |
| Model               | XGBoost               |
| Experiment tracking | MLflow                |
| Streaming           | Kafka (kafka-python)  |
| Feature store       | Redis                 |
| Explainability      | SHAP                  |
| Data science        | pandas, scikit-learn  |
| Packaging           | Docker                |

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/chandrahas748812-cmyk/realtime-fraud-detection-platform.git
cd realtime-fraud-detection-platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train a model (points at your labelled CSV)
python -m src.models.train --data data/transactions.csv

# 4. Start the inference API
uvicorn src.api.main:app --reload

# 5. Score a transaction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"amount": 1240.50, "merchant": "electronics-outlet",
       "latitude": 40.7128, "longitude": -74.0060,
       "card_last4": "4242", "merchant_category": "electronics"}'
```

Or run everything in Docker:

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## Project Layout

```text
.
+-- src/
|   +-- api/
|   |   +-- main.py          # FastAPI app: POST /score
|   +-- features/
|   |   +-- engineering.py   # Feature engineering functions
|   +-- models/
|       +-- train.py         # XGBoost training + MLflow logging
|   +-- monitoring/
|       +-- drift.py         # PSI drift checks vs baseline
+-- requirements.txt
+-- Dockerfile
+-- .gitignore
```

## API Contract

`POST /score` accepts a transaction JSON with `amount`, `merchant`,
`location` (lat/lon), `time` and card features, and responds with:

```json
{
  "fraud_probability": 0.87,
  "risk_tier": "high",
  "latency_ms": 42.5
}
```

Risk tiers: `low` (< 0.2), `medium` (0.2-0.7), `high` (>= 0.7).

## Portfolio note

Portfolio project demonstrating production MLOps patterns used in real-time
fraud detection systems.

*Disclaimer: this is a personal portfolio project. It is not affiliated
with any bank, and none of the metrics, figures or architecture details
here describe any real production system.*
