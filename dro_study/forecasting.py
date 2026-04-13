"""Constrained volatility forecasting and Heston parameter mapping."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import List, Sequence

from .config import ForecasterConfig
from .data import MarketObservation
from .regimes import one_hot_regime


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def _safe_stdev(values: Sequence[float], default: float = 0.0) -> float:
    return pstdev(values) if len(values) > 1 else default


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _softplus(value: float) -> float:
    if value > 20:
        return value
    return math.log1p(math.exp(value))


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2 or len(left) != len(right):
        return 0.0
    left_mean = _safe_mean(left)
    right_mean = _safe_mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(max(left_var * right_var, 1e-12))
    return numerator / denominator


def _enforce_feller(kappa: float, theta: float, sigma_v: float, margin: float = 0.98) -> float:
    max_sigma = math.sqrt(max(2.0 * kappa * theta * margin, 1e-10))
    return _clamp(sigma_v, 0.01, max_sigma)


@dataclass
class ForecastState:
    predicted_vol: float
    uncertainty: float
    mu: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    diagnostics: dict = field(default_factory=dict)


class SimpleMLP:
    """A tiny pure-Python neural network for portability."""

    def __init__(self, input_size: int, hidden_size: int, seed: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rng = random.Random(seed)
        scale = 1.0 / math.sqrt(max(input_size, 1))
        self.hidden_weights = [
            [self.rng.uniform(-0.5, 0.5) * scale for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.hidden_biases = [0.0 for _ in range(hidden_size)]
        self.output_weights = [self.rng.uniform(-0.5, 0.5) * scale for _ in range(hidden_size)]
        self.output_bias = 0.0

    def predict(self, features: Sequence[float]) -> float:
        hidden = []
        for weights, bias in zip(self.hidden_weights, self.hidden_biases):
            activation = sum(weight * value for weight, value in zip(weights, features)) + bias
            hidden.append(math.tanh(activation))
        return sum(weight * value for weight, value in zip(self.output_weights, hidden)) + self.output_bias

    def fit(self, samples: Sequence[Sequence[float]], targets: Sequence[float], epochs: int, lr: float) -> None:
        if not samples:
            return

        order = list(range(len(samples)))
        for _ in range(epochs):
            self.rng.shuffle(order)
            for index in order:
                features = samples[index]
                target = targets[index]

                hidden_raw = []
                hidden = []
                for weights, bias in zip(self.hidden_weights, self.hidden_biases):
                    activation = sum(weight * value for weight, value in zip(weights, features)) + bias
                    hidden_raw.append(activation)
                    hidden.append(math.tanh(activation))

                old_output_weights = list(self.output_weights)
                prediction = sum(weight * value for weight, value in zip(old_output_weights, hidden)) + self.output_bias
                error = _clamp(prediction - target, -4.0, 4.0)

                for hidden_index in range(self.hidden_size):
                    grad_out = error * hidden[hidden_index]
                    self.output_weights[hidden_index] -= lr * grad_out
                self.output_bias -= lr * error

                for hidden_index in range(self.hidden_size):
                    delta = error * old_output_weights[hidden_index] * (1.0 - hidden[hidden_index] ** 2)
                    delta = _clamp(delta, -4.0, 4.0)
                    for feature_index in range(self.input_size):
                        self.hidden_weights[hidden_index][feature_index] -= lr * delta * features[feature_index]
                    self.hidden_biases[hidden_index] -= lr * delta


class EnsembleVolatilityForecaster:
    """
    Forecast next-period realized volatility and convert it into Heston parameters.

    This addresses the project's gaps by moving from direct return-point prediction to
    volatility-state estimation, predictive uncertainty, and structurally valid parameters.
    """

    def __init__(self, config: ForecasterConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed
        self.models: List[SimpleMLP] = []
        self.feature_mean: List[float] = []
        self.feature_std: List[float] = []
        self.target_mean: float = 0.0
        self.target_std: float = 1.0
        self.training_metrics: dict = {}
        self._default_log_vol: float = math.log(0.18)

    def _feature_vector(self, history: Sequence[MarketObservation]) -> List[float]:
        returns = [item.log_return for item in history]
        vols = [max(item.realized_vol, 0.05) for item in history]
        vix_levels = [item.vix / 100.0 for item in history]
        abs_returns = [abs(item.log_return) for item in history]
        short_returns = returns[-5:]
        short_vols = vols[-5:]
        vol_changes = [vols[index] - vols[index - 1] for index in range(1, len(vols))]
        aligned_returns = returns[1:]
        leverage_corr = _correlation(aligned_returns[-len(vol_changes) :], vol_changes[-len(aligned_returns) :])

        features = [
            returns[-1] if returns else 0.0,
            _safe_mean(short_returns),
            _safe_mean(returns),
            _safe_stdev(short_returns),
            _safe_stdev(returns),
            _safe_mean(abs_returns[-5:]),
            _safe_mean(abs_returns),
            vols[-1] if vols else 0.18,
            _safe_mean(short_vols, 0.18),
            _safe_mean(vols, 0.18),
            (vols[-1] - _safe_mean(vols, vols[-1])) if vols else 0.0,
            _safe_stdev(vol_changes),
            vix_levels[-1] if vix_levels else 0.20,
            _safe_mean(vix_levels[-5:], 0.20),
            (vix_levels[-1] - vix_levels[max(0, len(vix_levels) - 5)]) if len(vix_levels) > 1 else 0.0,
            history[-1].inflation_change if history else 0.0,
            history[-1].rate_change if history else 0.0,
            sum(1.0 for item in returns[-5:] if item < 0.0) / max(len(short_returns), 1),
            leverage_corr,
        ]
        features.extend(one_hot_regime(history[-1].regime if history else "low_inflation_easing"))
        return features

    def _build_dataset(self, observations: Sequence[MarketObservation]) -> tuple[List[List[float]], List[float]]:
        samples: List[List[float]] = []
        targets: List[float] = []
        lookback = self.config.lookback
        if len(observations) <= lookback:
            return samples, targets

        for end_index in range(lookback - 1, len(observations) - 1):
            history = observations[end_index - lookback + 1 : end_index + 1]
            next_obs = observations[end_index + 1]
            samples.append(self._feature_vector(history))
            targets.append(math.log(max(next_obs.realized_vol, 0.05)))
        return samples, targets

    def _fit_standardization(self, samples: Sequence[Sequence[float]], targets: Sequence[float]) -> None:
        if not samples:
            self.feature_mean = []
            self.feature_std = []
            self.target_mean = self._default_log_vol
            self.target_std = 1.0
            return

        dimensions = len(samples[0])
        self.feature_mean = [
            _safe_mean([sample[index] for sample in samples]) for index in range(dimensions)
        ]
        self.feature_std = [
            max(_safe_stdev([sample[index] for sample in samples]), 1e-6) for index in range(dimensions)
        ]
        self.target_mean = _safe_mean(targets, self._default_log_vol)
        self.target_std = max(_safe_stdev(targets), 1e-6)

    def _standardize_features(self, sample: Sequence[float]) -> List[float]:
        return [
            (value - mean_value) / std_value
            for value, mean_value, std_value in zip(sample, self.feature_mean, self.feature_std)
        ]

    def _destandardize_target(self, value: float) -> float:
        return self.target_mean + self.target_std * value

    def fit(self, observations: Sequence[MarketObservation]) -> None:
        samples, targets = self._build_dataset(observations)
        self._fit_standardization(samples, targets)
        if not samples:
            self.models = []
            self.training_metrics = {}
            return

        standardized_samples = [self._standardize_features(sample) for sample in samples]
        standardized_targets = [
            (target - self.target_mean) / self.target_std for target in targets
        ]
        self.models = []
        base_rng = random.Random(self.seed)

        for model_index in range(self.config.ensemble_size):
            bootstrap_ids = [
                base_rng.randrange(len(standardized_samples)) for _ in range(len(standardized_samples))
            ]
            model = SimpleMLP(
                input_size=len(standardized_samples[0]),
                hidden_size=self.config.hidden_size,
                seed=self.seed + 97 * (model_index + 1),
            )
            model.fit(
                [standardized_samples[item] for item in bootstrap_ids],
                [standardized_targets[item] for item in bootstrap_ids],
                epochs=self.config.epochs,
                lr=self.config.learning_rate,
            )
            self.models.append(model)

        predictions = []
        for sample in standardized_samples:
            model_values = [model.predict(sample) for model in self.models]
            predictions.append(self._destandardize_target(_safe_mean(model_values)))

        rmse = math.sqrt(
            _safe_mean([(prediction - target) ** 2 for prediction, target in zip(predictions, targets)], 0.0)
        )
        mae = _safe_mean([abs(prediction - target) for prediction, target in zip(predictions, targets)], 0.0)
        self.training_metrics = {"rmse_log_vol": rmse, "mae_log_vol": mae}

    def predict(self, history: Sequence[MarketObservation]) -> ForecastState:
        if not history:
            return ForecastState(
                predicted_vol=0.18,
                uncertainty=0.10,
                mu=0.05,
                kappa=2.5,
                theta=0.18**2,
                sigma_v=0.18,
                rho=-0.5,
                diagnostics={"fallback": True},
            )

        features = self._feature_vector(history)
        log_vol_predictions: List[float] = []
        if self.models and self.feature_mean:
            standardized = self._standardize_features(features)
            log_vol_predictions = [
                self._destandardize_target(model.predict(standardized)) for model in self.models
            ]

        if log_vol_predictions:
            log_vol = _safe_mean(log_vol_predictions)
            log_vol_uncertainty = _safe_stdev(log_vol_predictions)
        else:
            log_vol = math.log(max(history[-1].realized_vol, 0.12))
            fallback_vols = [item.realized_vol for item in history[-5:]]
            log_vol_uncertainty = _safe_stdev([math.log(max(item, 0.05)) for item in fallback_vols], 0.08)

        predicted_vol = _clamp(math.exp(log_vol), 0.05, 0.95)
        uncertainty = _clamp(abs(log_vol_uncertainty), 0.01, 0.35)

        recent_returns = [item.log_return for item in history]
        recent_vols = [max(item.realized_vol, 0.05) for item in history]
        vol_changes = [recent_vols[index] - recent_vols[index - 1] for index in range(1, len(recent_vols))]

        recent_mean_return = _safe_mean(recent_returns[-5:])
        long_mean_return = _safe_mean(recent_returns)
        # Keep the neural drift estimate informative but not dominant.
        mu_signal = 0.4 * recent_mean_return + 0.6 * long_mean_return
        mu = _clamp(mu_signal * 252.0, -0.08, 0.15)

        long_vol = _safe_mean(recent_vols, predicted_vol)
        vol_gap = abs(predicted_vol - long_vol) / max(long_vol, 1e-6)
        kappa = 0.8 + 4.4 * _sigmoid(4.0 * vol_gap + 2.0 * uncertainty)
        theta = max(predicted_vol * predicted_vol, 1e-5)

        vol_of_vol = _safe_stdev(vol_changes, predicted_vol * 0.10) * math.sqrt(252.0)
        sigma_v_candidate = _softplus(0.04 + 0.9 * vol_of_vol + 0.30 * uncertainty)
        sigma_v = _enforce_feller(kappa=kappa, theta=theta, sigma_v=sigma_v_candidate)

        leverage_corr = _correlation(recent_returns[1:], vol_changes) if len(recent_returns) > 2 else -0.25
        tightening_penalty = 0.10 if history[-1].regime.endswith("tightening") else 0.0
        rho = _clamp(-0.30 - 0.35 * leverage_corr - tightening_penalty, -0.95, 0.20)

        diagnostics = {
            "predicted_log_vol": log_vol,
            "training": self.training_metrics,
            "feller_gap": 2.0 * kappa * theta - sigma_v * sigma_v,
        }
        return ForecastState(
            predicted_vol=predicted_vol,
            uncertainty=uncertainty,
            mu=mu,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            diagnostics=diagnostics,
        )
