"""Data loading and feature engineering utilities."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, List, Optional


@dataclass
class MarketObservation:
    date: date
    close: float
    vix: float
    cpi: float
    fed_funds: float
    log_return: float = 0.0
    simple_return: float = 0.0
    realized_vol: float = 0.15
    inflation_change: float = 0.0
    rate_change: float = 0.0
    regime: str = ""


def _parse_date(raw: str) -> date:
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {raw!r}")


def _first_numeric(row: dict, keys: Iterable[str], default: Optional[float] = None) -> float:
    lower = {str(key).strip().lower(): value for key, value in row.items()}
    for key in keys:
        if key in lower and str(lower[key]).strip() not in {"", "None", "nan"}:
            return float(lower[key])
    if default is None:
        raise KeyError(f"Expected one of {tuple(keys)} in CSV row.")
    return float(default)


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def _safe_stdev(values: List[float], default: float = 0.0) -> float:
    return pstdev(values) if len(values) > 1 else default


def prepare_observations(
    observations: List[MarketObservation],
    realized_vol_window: int = 21,
    macro_window: int = 21,
) -> List[MarketObservation]:
    """Compute returns, realized volatility, and macro deltas in-place."""
    if not observations:
        return observations

    observations.sort(key=lambda obs: obs.date)
    previous_close = observations[0].close

    for index, obs in enumerate(observations):
        if index == 0 or previous_close <= 0:
            obs.log_return = 0.0
            obs.simple_return = 0.0
        else:
            ratio = max(obs.close / previous_close, 1e-9)
            obs.log_return = math.log(ratio)
            obs.simple_return = ratio - 1.0
        previous_close = obs.close

        start = max(0, index - realized_vol_window + 1)
        window_returns = [item.log_return for item in observations[start : index + 1]]
        obs.realized_vol = _safe_stdev(window_returns, 0.15 / math.sqrt(252.0)) * math.sqrt(252.0)
        obs.realized_vol = max(obs.realized_vol, 0.05)

        macro_anchor = max(0, index - macro_window)
        past_cpi = observations[macro_anchor].cpi
        past_rate = observations[macro_anchor].fed_funds
        obs.inflation_change = ((obs.cpi - past_cpi) / past_cpi) if past_cpi else 0.0
        obs.rate_change = obs.fed_funds - past_rate

    return observations


def load_market_csv(path: str | Path) -> List[MarketObservation]:
    """Load SPY-like data from CSV using a forgiving column schema."""
    rows: List[MarketObservation] = []
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        last_vix = 20.0
        last_cpi = 100.0
        last_rate = 2.0
        for raw in reader:
            current_vix = _first_numeric(raw, ("vix", "vvix", "implied_vol"), last_vix)
            current_cpi = _first_numeric(raw, ("cpi", "inflation_index"), last_cpi)
            current_rate = _first_numeric(raw, ("fed_funds", "fedfunds", "policy_rate"), last_rate)

            rows.append(
                MarketObservation(
                    date=_parse_date(str(raw.get("date", raw.get("Date", "")))),
                    close=_first_numeric(raw, ("close", "adj_close", "adjclose", "spy", "price")),
                    vix=current_vix,
                    cpi=current_cpi,
                    fed_funds=current_rate,
                )
            )
            last_vix = current_vix
            last_cpi = current_cpi
            last_rate = current_rate

    return prepare_observations(rows)


def generate_synthetic_market_data(days: int = 650, seed: int = 7) -> List[MarketObservation]:
    """
    Create a synthetic dataset so the full project pipeline runs without external downloads.

    The synthetic process cycles through four macro regimes and produces price, volatility,
    inflation, and rate series that are coherent enough for backtesting smoke tests.
    """

    rng = random.Random(seed)
    start_date = date(2000, 1, 3)
    close = 100.0
    cpi = 100.0
    fed_funds = 3.0
    state = "low_inflation_tightening"

    regime_specs = {
        "low_inflation_easing": {"mu": 0.13, "vol": 0.14, "infl": 0.018, "rate_step": -0.0009},
        "low_inflation_tightening": {"mu": 0.09, "vol": 0.17, "infl": 0.022, "rate_step": 0.0008},
        "high_inflation_easing": {"mu": 0.05, "vol": 0.21, "infl": 0.042, "rate_step": -0.0012},
        "high_inflation_tightening": {"mu": 0.01, "vol": 0.30, "infl": 0.060, "rate_step": 0.0013},
    }
    transitions = {
        "low_inflation_easing": (
            ("low_inflation_easing", 0.80),
            ("low_inflation_tightening", 0.12),
            ("high_inflation_easing", 0.05),
            ("high_inflation_tightening", 0.03),
        ),
        "low_inflation_tightening": (
            ("low_inflation_tightening", 0.78),
            ("low_inflation_easing", 0.10),
            ("high_inflation_tightening", 0.09),
            ("high_inflation_easing", 0.03),
        ),
        "high_inflation_easing": (
            ("high_inflation_easing", 0.74),
            ("high_inflation_tightening", 0.12),
            ("low_inflation_easing", 0.09),
            ("low_inflation_tightening", 0.05),
        ),
        "high_inflation_tightening": (
            ("high_inflation_tightening", 0.82),
            ("high_inflation_easing", 0.08),
            ("low_inflation_tightening", 0.07),
            ("low_inflation_easing", 0.03),
        ),
    }

    rows: List[MarketObservation] = []
    current_date = start_date
    for _ in range(days):
        draw = rng.random()
        running = 0.0
        for next_state, probability in transitions[state]:
            running += probability
            if draw <= running:
                state = next_state
                break

        spec = regime_specs[state]
        daily_vol = spec["vol"] / math.sqrt(252.0)
        daily_drift = spec["mu"] / 252.0
        shock = rng.gauss(0.0, 1.0)
        log_return = daily_drift - 0.5 * daily_vol * daily_vol + daily_vol * shock
        close *= math.exp(log_return)

        vix = max(10.0, spec["vol"] * 100.0 + rng.gauss(0.0, 2.5))
        cpi *= 1.0 + spec["infl"] / 252.0 + rng.gauss(0.0, 0.00008)
        fed_funds = max(0.0, fed_funds + spec["rate_step"] + rng.gauss(0.0, 0.00035))

        rows.append(
            MarketObservation(
                date=current_date,
                close=close,
                vix=vix,
                cpi=cpi,
                fed_funds=fed_funds,
            )
        )
        current_date += timedelta(days=1)

    return prepare_observations(rows)
