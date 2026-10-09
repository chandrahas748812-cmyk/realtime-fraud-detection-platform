"""Feature engineering for real-time fraud scoring.

Pure functions that turn a raw transaction plus recent history (as served
by the Redis feature store) into the numeric feature vector consumed by
the XGBoost model. Keeping these functions side-effect free makes them
reusable in training, batch backfills and the streaming path.
"""

from __future__ import annotations

import math
from datetime import datetime

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def velocity_features(amounts_24h: list[float]) -> dict:
    """Transaction velocity features over the trailing 24 hours.

    Args:
        amounts_24h: amounts of the user's transactions in the last 24h,
            most recent last (excluding the current transaction).

    Returns:
        Dict with ``txn_count_24h``, ``total_amount_24h`` and
        ``avg_amount_24h``.
    """
    count = len(amounts_24h)
    total = float(sum(amounts_24h)) if count else 0.0
    return {
        "txn_count_24h": count,
        "total_amount_24h": round(total, 2),
        "avg_amount_24h": round(total / count, 2) if count else 0.0,
    }


def amount_zscore(amount: float, history: list[float]) -> float:
    """Z-score of ``amount`` against the user's historical amounts.

    A large positive z-score means the transaction is unusually large
    for this user -- a classic fraud signal. Returns 0.0 when there is
    not enough history (fewer than 2 prior transactions).
    """
    if len(history) < 2:
        return 0.0
    mean = sum(history) / len(history)
    variance = sum((x - mean) ** 2 for x in history) / (len(history) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return round((amount - mean) / std, 4)


def geo_distance_km(
    latitude: float,
    longitude: float,
    last_latitude: float | None,
    last_longitude: float | None,
) -> float:
    """Distance from the user's previous transaction location.

    Impossible-travel (e.g. two distant cities minutes apart) is a
    strong fraud indicator. Returns 0.0 when no prior location exists.
    """
    if last_latitude is None or last_longitude is None:
        return 0.0
    return round(
        haversine_km(latitude, longitude, last_latitude, last_longitude), 3
    )


def time_of_day_risk(timestamp: str) -> float:
    """Risk weight for the transaction's local time of day.

    Card-present fraud skews towards late-night hours; this returns a
    small additive risk bump in [0, 1] that the model can combine with
    stronger signals. Accepts an ISO-8601 timestamp string.
    """
    try:
        hour = datetime.fromisoformat(timestamp).hour
    except ValueError:
        return 0.0
    if 0 <= hour < 5:
        return 0.8  # deep night
    if 5 <= hour < 7:
        return 0.3  # early morning
    return 0.0


def merchant_risk_score(
    merchant_category: str, category_fraud_rates: dict | None = None
) -> float:
    """Prior fraud rate for a merchant category, in [0, 1].

    ``category_fraud_rates`` maps categories to historical fraud rates
    (computed offline). Unknown categories fall back to the global
    prior of 0.02.
    """
    rates = category_fraud_rates or {}
    return float(rates.get(merchant_category, 0.02))


def build_feature_vector(
    amount: float,
    latitude: float,
    longitude: float,
    timestamp: str,
    merchant_category: str,
    amounts_24h: list[float] | None = None,
    amount_history: list[float] | None = None,
    last_latitude: float | None = None,
    last_longitude: float | None = None,
    category_fraud_rates: dict | None = None,
) -> dict:
    """Assemble the full numeric feature vector for one transaction.

    This is the single entry point used by the API at scoring time and
    by the training script when materialising the training set.
    """
    amounts_24h = amounts_24h or []
    amount_history = amount_history or []
    features = {
        "amount": amount,
        "amount_zscore": amount_zscore(amount, amount_history),
        "geo_distance_km": geo_distance_km(
            latitude, longitude, last_latitude, last_longitude
        ),
        "time_of_day_risk": time_of_day_risk(timestamp),
        "merchant_risk_score": merchant_risk_score(
            merchant_category, category_fraud_rates
        ),
    }
    features.update(velocity_features(amounts_24h))
    return features
