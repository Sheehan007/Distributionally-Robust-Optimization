"""Scenario engines for Heston, LSTM-only, and hybrid return generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List

from .config import RegimePrior, SimulationConfig
from .forecasting import ForecastState


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def _tail_mean(returns: List[float], alpha: float = 0.95) -> float:
    if not returns:
        return 0.0
    tail_count = max(1, int(math.ceil(len(returns) * (1.0 - alpha))))
    sorted_returns = sorted(returns)
    return _safe_mean(sorted_returns[:tail_count])


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _skewness(values: List[float]) -> float:
    if len(values) < 3:
        return 0.0
    mu = _safe_mean(values)
    centered = [value - mu for value in values]
    variance = _safe_mean([value * value for value in centered], 0.0)
    if variance <= 1e-12:
        return 0.0
    third = _safe_mean([value ** 3 for value in centered], 0.0)
    return third / (variance ** 1.5)


def summarize_returns(returns: List[float]) -> Dict[str, float]:
    average = _safe_mean(returns)
    return {
        "mean": average,
        "tail_cvar_95": _tail_mean(returns, alpha=0.95),
        "skewness": _skewness(returns),
        "variance": _safe_mean([(item - average) ** 2 for item in returns], 0.0),
        "q05": sorted(returns)[max(0, int(len(returns) * 0.05) - 1)] if returns else 0.0,
    }


@dataclass
class ScenarioBundle:
    model_name: str
    returns: List[float]
    summary: Dict[str, float] = field(default_factory=dict)


class RegimeAwareHestonEngine:
    """Blend neural state estimates with regime priors and generate comparable baselines."""

    def __init__(self, config: SimulationConfig, regime_priors: Dict[str, RegimePrior]) -> None:
        self.config = config
        self.regime_priors = regime_priors

    def _blended_parameters(self, forecast: ForecastState, regime: str) -> RegimePrior:
        prior = self.regime_priors[regime]
        theta = max(0.55 * prior.theta + 0.45 * forecast.theta * (1.0 + forecast.uncertainty), 1e-5)
        kappa = 0.55 * prior.kappa + 0.45 * forecast.kappa
        sigma_v = min(
            math.sqrt(max(2.0 * kappa * theta * 0.98, 1e-10)),
            (0.55 * prior.sigma_v + 0.45 * forecast.sigma_v) * (1.0 + 0.8 * forecast.uncertainty),
        )
        rho = max(-0.95, min(0.95, 0.6 * prior.rho + 0.4 * forecast.rho))
        mu = _clamp(0.75 * prior.mu + 0.25 * forecast.mu, -0.08, 0.16)
        return RegimePrior(mu=mu, kappa=kappa, theta=theta, sigma_v=max(0.01, sigma_v), rho=rho)

    def _simulate_heston_bundle(
        self,
        params: RegimePrior,
        initial_variance: float,
        seed: int,
        model_name: str,
        current_price: float = 1.0,
        horizon_days: int | None = None,
    ) -> ScenarioBundle:
        horizon = self.config.horizon_days if horizon_days is None else horizon_days
        dt = self.config.dt
        sqrt_dt = math.sqrt(dt)
        rng = random.Random(seed)

        returns: List[float] = []
        for _ in range(self.config.scenario_count):
            price = current_price
            variance = max(initial_variance, 1e-8)
            for _ in range(horizon):
                shock_price = rng.gauss(0.0, 1.0)
                shock_independent = rng.gauss(0.0, 1.0)
                shock_var = params.rho * shock_price + math.sqrt(max(1.0 - params.rho * params.rho, 1e-10)) * shock_independent

                sqrt_variance = math.sqrt(max(variance, 1e-10))
                variance = max(
                    variance
                    + params.kappa * (params.theta - variance) * dt
                    + params.sigma_v * sqrt_variance * sqrt_dt * shock_var
                    + 0.25 * params.sigma_v * params.sigma_v * dt * (shock_var * shock_var - 1.0),
                    1e-8,
                )
                log_step_return = (params.mu - 0.5 * variance) * dt + math.sqrt(variance) * sqrt_dt * shock_price
                log_step_return = _clamp(
                    log_step_return,
                    -self.config.clamp_log_return,
                    self.config.clamp_log_return,
                )
                price *= math.exp(log_step_return)

            returns.append(price / current_price - 1.0)

        return ScenarioBundle(model_name=model_name, returns=returns, summary=summarize_returns(returns))

    def simulate_hybrid_returns(
        self,
        current_price: float,
        forecast: ForecastState,
        regime: str,
        seed: int,
        horizon_days: int | None = None,
    ) -> ScenarioBundle:
        params = self._blended_parameters(forecast, regime)
        initial_variance = max(params.theta * (1.0 + 0.5 * forecast.uncertainty), 1e-5)
        return self._simulate_heston_bundle(
            params=params,
            initial_variance=initial_variance,
            seed=seed,
            model_name="hybrid",
            current_price=current_price,
            horizon_days=horizon_days,
        )

    def simulate_returns(
        self,
        current_price: float,
        forecast: ForecastState,
        regime: str,
        seed: int,
        horizon_days: int | None = None,
    ) -> ScenarioBundle:
        return self.simulate_hybrid_returns(current_price, forecast, regime, seed, horizon_days)

    def simulate_static_heston_returns(
        self,
        current_price: float,
        realized_vol: float,
        regime: str,
        seed: int,
        horizon_days: int | None = None,
    ) -> ScenarioBundle:
        params = self.regime_priors[regime]
        initial_variance = max(realized_vol * realized_vol, params.theta)
        return self._simulate_heston_bundle(
            params=params,
            initial_variance=initial_variance,
            seed=seed,
            model_name="static_heston",
            current_price=current_price,
            horizon_days=horizon_days,
        )

    def simulate_lstm_only_returns(
        self,
        forecast: ForecastState,
        regime: str,
        seed: int,
        horizon_days: int | None = None,
    ) -> ScenarioBundle:
        horizon = self.config.horizon_days if horizon_days is None else horizon_days
        dt = self.config.dt
        sqrt_dt = math.sqrt(dt)
        rng = random.Random(seed)
        prior = self.regime_priors[regime]
        mu = _clamp(0.60 * prior.mu + 0.40 * forecast.mu, -0.08, 0.16)

        returns: List[float] = []
        for _ in range(self.config.scenario_count):
            price = 1.0
            scenario_vol = _clamp(
                forecast.predicted_vol * math.exp(rng.gauss(0.0, max(forecast.uncertainty, 0.01))),
                0.05,
                0.95,
            )
            for _ in range(horizon):
                shock = rng.gauss(0.0, 1.0)
                log_step_return = (mu - 0.5 * scenario_vol * scenario_vol) * dt + scenario_vol * sqrt_dt * shock
                log_step_return = _clamp(
                    log_step_return,
                    -self.config.clamp_log_return,
                    self.config.clamp_log_return,
                )
                price *= math.exp(log_step_return)
            returns.append(price - 1.0)

        return ScenarioBundle(model_name="lstm_only", returns=returns, summary=summarize_returns(returns))
