"""Train the fraud-detection XGBoost classifier.

Loads a labelled transaction CSV, builds the feature matrix with
``src.features.engineering``, trains an XGBoost classifier with class
weighting for the imbalanced fraud problem, evaluates it, and logs
parameters, metrics and the model artefact to MLflow.

Usage:
    python -m src.models.train --data data/transactions.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.engineering import build_feature_vector

# Feature columns in the order the model expects them.
FEATURE_COLUMNS = [
    "amount",
    "amount_zscore",
    "geo_distance_km",
    "time_of_day_risk",
    "merchant_risk_score",
    "txn_count_24h",
    "total_amount_24h",
    "avg_amount_24h",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train the fraud detection XGBoost model."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ.get("FRAUD_DATA_PATH", "data/transactions.csv"),
        help="Path to the labelled transactions CSV (configurable).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save the trained model artefact.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_dataset(path: str) -> pd.DataFrame:
    """Load the labelled transaction dataset.

    Expected columns: ``amount, latitude, longitude, timestamp,
    merchant_category, label`` where ``label`` is 1 for fraud.
    Optional columns (used when present): ``amounts_24h``,
    ``amount_history``, ``last_latitude``, ``last_longitude``.
    """
    df = pd.read_csv(path)
    required = {
        "amount",
        "latitude",
        "longitude",
        "timestamp",
        "merchant_category",
        "label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering row-wise to the raw transactions."""
    rows = []
    for _, row in df.iterrows():
        rows.append(
            build_feature_vector(
                amount=float(row["amount"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                timestamp=str(row["timestamp"]),
                merchant_category=str(row["merchant_category"]),
            )
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def train(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Fit the XGBoost classifier with class weighting.

    ``scale_pos_weight`` compensates for the heavy class imbalance
    typical of fraud data (far fewer frauds than legit transactions).
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = (neg / pos) if pos else 1.0
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test) -> dict:
    """Compute PR-AUC and precision/recall at the alert threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    return {
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "precision_at_50": round(float(precision[50]), 4)
        if len(precision) > 50
        else None,
        "recall_at_50": round(float(recall[50]), 4)
        if len(recall) > 50
        else None,
    }


def main() -> None:
    """Train, evaluate, log to MLflow and persist the model."""
    args = parse_args()
    df = load_dataset(args.data)
    X = featurize(df)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state,
        stratify=y,
    )

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run():
        model = train(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_params({
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "test_size": args.test_size,
        })
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if v is not None}
        )
        mlflow.xgboost.log_model(model, artifact_path="model")
        print(f"Metrics: {metrics}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "model.json"))
    print(f"Model artefact saved to {output_dir / 'model.json'}")


if __name__ == "__main__":
    main()
