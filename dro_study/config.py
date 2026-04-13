"""Configuration objects for the DRO study pipeline."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ForecasterConfig:
    lookback: int = 20
    hidden_size: int = 10
    epochs: int = 60
    learning_rate: float = 0.01
    ensemble_size: int = 3
    seed: int = 7


@dataclass
class SimulationConfig:
    scenario_count: int = 250
    horizon_days: int = 5
    dt: float = 1.0 / 252.0
    clamp_log_return: float = 0.18


@dataclass
class OptimizerConfig:
    alpha: float = 0.95
    risk_aversion: float = 2.5
    variance_penalty: float = 4.0
    epsilon_base: float = 0.005
    fixed_epsilon: float = 0.02
    uncertainty_scale: float = 0.08
    turnover_cost_bps: float = 7.0
    max_weight: float = 1.0
    weight_step: float = 0.05
    cash_rate: float = 0.03
    regime_penalties: Dict[str, float] = field(
        default_factory=lambda: {
            "low_inflation_easing": 0.000,
            "low_inflation_tightening": 0.004,
            "high_inflation_easing": 0.006,
            "high_inflation_tightening": 0.012,
        }
    )


@dataclass
class BacktestConfig:
    train_window: int = 126
    rebalance_every: int = 5


@dataclass
class RegimePrior:
    mu: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float


def default_regime_priors() -> Dict[str, RegimePrior]:
    return {
        "low_inflation_easing": RegimePrior(mu=0.12, kappa=2.6, theta=0.020, sigma_v=0.20, rho=-0.35),
        "low_inflation_tightening": RegimePrior(mu=0.08, kappa=3.0, theta=0.030, sigma_v=0.24, rho=-0.45),
        "high_inflation_easing": RegimePrior(mu=0.05, kappa=3.3, theta=0.045, sigma_v=0.28, rho=-0.58),
        "high_inflation_tightening": RegimePrior(mu=0.01, kappa=3.8, theta=0.065, sigma_v=0.34, rho=-0.72),
    }


@dataclass
class StudyConfig:
    forecaster: ForecasterConfig = field(default_factory=ForecasterConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    regime_priors: Dict[str, RegimePrior] = field(default_factory=default_regime_priors)


def default_study_config() -> StudyConfig:
    """Return a fresh study configuration."""
    return StudyConfig()
