"""Macro regime classification helpers."""

from __future__ import annotations

from statistics import median
from typing import Iterable, Optional, Tuple

from .data import MarketObservation


REGIME_ORDER = (
    "low_inflation_easing",
    "low_inflation_tightening",
    "high_inflation_easing",
    "high_inflation_tightening",
)

REGIME_LABELS = {
    "low_inflation_easing": "Low inflation / easing",
    "low_inflation_tightening": "Low inflation / tightening",
    "high_inflation_easing": "High inflation / easing",
    "high_inflation_tightening": "High inflation / tightening",
}


def infer_thresholds(observations: Iterable[MarketObservation]) -> Tuple[float, float]:
    obs = list(observations)
    inflation_threshold = median(item.inflation_change for item in obs) if obs else 0.0
    rate_threshold = median(item.rate_change for item in obs) if obs else 0.0
    return inflation_threshold, rate_threshold


def classify_regime(
    inflation_change: float,
    rate_change: float,
    inflation_threshold: float,
    rate_threshold: float,
) -> str:
    high_inflation = inflation_change >= inflation_threshold
    tightening = rate_change >= rate_threshold
    if high_inflation and tightening:
        return "high_inflation_tightening"
    if high_inflation and not tightening:
        return "high_inflation_easing"
    if not high_inflation and tightening:
        return "low_inflation_tightening"
    return "low_inflation_easing"


def attach_regimes(
    observations: Iterable[MarketObservation],
    inflation_threshold: Optional[float] = None,
    rate_threshold: Optional[float] = None,
) -> Tuple[float, float]:
    obs = list(observations)
    inferred_inflation, inferred_rate = infer_thresholds(obs)
    inflation_cut = inferred_inflation if inflation_threshold is None else inflation_threshold
    rate_cut = inferred_rate if rate_threshold is None else rate_threshold

    for item in obs:
        item.regime = classify_regime(
            inflation_change=item.inflation_change,
            rate_change=item.rate_change,
            inflation_threshold=inflation_cut,
            rate_threshold=rate_cut,
        )

    return inflation_cut, rate_cut


def one_hot_regime(regime: str) -> list[float]:
    return [1.0 if regime == item else 0.0 for item in REGIME_ORDER]
