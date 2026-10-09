"""Drift monitoring for the fraud-detection feature pipeline.

Compares recent (production) feature distributions against the training
baseline using the Population Stability Index (PSI):

- PSI < 0.10: no significant population shift
- 0.10 - 0.25: moderate shift, investigate
- PSI > 0.25: significant shift, consider retraining

Usage:
    python -m src.monitoring.drift --baseline data/baseline.csv \
        --recent data/recent.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

PSI_NO_CHANGE = 0.1
PSI_MODERATE_SHIFT = 0.25


def calculate_psi(
    expected: pd.Series, actual: pd.Series, buckets: int = 10
) -> float:
    """Population Stability Index between baseline and recent values.

    Args:
        expected: baseline (training) feature values.
        actual: recent (production) feature values.
        buckets: number of quantile buckets for binning.

    Returns:
        PSI score; higher means more drift.
    """
    breakpoints = [
        expected.quantile(i / buckets) for i in range(1, buckets)
    ]
    # Guard against duplicate edges on constant features.
    breakpoints = sorted(set(breakpoints))
    expected_counts = np.histogram(expected, bins=[-np.inf, *breakpoints, np.inf])[0]
    actual_counts = np.histogram(actual, bins=[-np.inf, *breakpoints, np.inf])[0]
    expected_percents = expected_counts / max(len(expected), 1)
    actual_percents = actual_counts / max(len(actual), 1)
    # Smooth zeros so the log ratio stays finite.
    expected_percents = np.where(expected_percents == 0, 1e-4, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-4, actual_percents)
    psi_values = (actual_percents - expected_percents) * np.log(
        actual_percents / expected_percents
    )
    return round(float(np.sum(psi_values)), 4)


def interpret_psi(psi: float) -> str:
    """Human-readable drift verdict for a PSI score."""
    if psi < PSI_NO_CHANGE:
        return "no significant shift"
    if psi < PSI_MODERATE_SHIFT:
        return "moderate shift - investigate"
    return "significant shift - consider retraining"


def check_drift(
    baseline: pd.DataFrame, recent: pd.DataFrame, features: list[str]
) -> dict:
    """Run PSI drift checks for each feature in ``features``.

    Returns a dict mapping feature name to ``{"psi", "verdict"}``.
    """
    report = {}
    for feature in features:
        if feature not in baseline.columns or feature not in recent.columns:
            report[feature] = {"psi": None, "verdict": "missing column"}
            continue
        psi = calculate_psi(baseline[feature], recent[feature])
        report[feature] = {"psi": psi, "verdict": interpret_psi(psi)}
    return report


def main() -> None:
    """CLI: compare baseline vs recent CSVs and print the drift report."""
    parser = argparse.ArgumentParser(description="PSI drift check.")
    parser.add_argument("--baseline", required=True, help="Baseline CSV path.")
    parser.add_argument("--recent", required=True, help="Recent CSV path.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["amount", "amount_zscore", "geo_distance_km"],
    )
    args = parser.parse_args()
    baseline = pd.read_csv(args.baseline)
    recent = pd.read_csv(args.recent)
    report = check_drift(baseline, recent, args.features)
    drifted = [f for f, r in report.items() if (r["psi"] or 0) >= PSI_NO_CHANGE]
    for feature, result in report.items():
        print(f"{feature}: PSI={result['psi']} ({result['verdict']})")
    if drifted:
        print(f"\nDrift detected in: {', '.join(drifted)}")
    else:
        print("\nNo significant drift detected.")


if __name__ == "__main__":
    main()
