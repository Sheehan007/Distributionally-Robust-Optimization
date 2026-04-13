"""Rolling backtesting engine for baselines, ablations, and DRO strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import StudyConfig
from .data import MarketObservation
from .dro import (
    adaptive_epsilon,
    optimize_mean_cvar_allocation,
    optimize_mean_variance_allocation,
)
from .forecasting import EnsembleVolatilityForecaster
from .heston import RegimeAwareHestonEngine, summarize_returns
from .regimes import attach_regimes


DEFAULT_STRATEGIES: Tuple[str, ...] = (
    "mean_variance",
    "vanilla_cvar",
    "static_heston_cvar",
    "lstm_only_cvar",
    "hybrid_cvar",
    "fixed_dro",
    "adaptive_dro",
)

STRATEGY_SPECS = {
    "mean_variance": {"objective_mode": "mean_variance", "scenario_source": "historical"},
    "vanilla_cvar": {"objective_mode": "empirical_cvar", "scenario_source": "historical"},
    "static_heston_cvar": {"objective_mode": "empirical_cvar", "scenario_source": "static_heston"},
    "lstm_only_cvar": {"objective_mode": "empirical_cvar", "scenario_source": "lstm_only"},
    "hybrid_cvar": {"objective_mode": "empirical_cvar", "scenario_source": "hybrid"},
    "fixed_dro": {"objective_mode": "wasserstein_dro", "scenario_source": "hybrid", "epsilon_mode": "fixed"},
    "adaptive_dro": {"objective_mode": "wasserstein_dro", "scenario_source": "hybrid", "epsilon_mode": "adaptive"},
}


@dataclass
class TradeRecord:
    date: str
    strategy: str
    scenario_source: str
    objective_mode: str
    regime: str
    risky_weight: float
    cash_weight: float
    epsilon: float
    eta: float
    forecast_vol: float
    uncertainty: float
    expected_return: float
    empirical_cvar_loss: float
    robust_cvar_loss: float
    realized_return: float
    turnover: float
    scenario_mean: float
    scenario_q05: float
    training_rmse_log_vol: float

    def to_dict(self) -> dict:
        return asdict(self)


def _compound_return(returns: Sequence[float]) -> float:
    wealth = 1.0
    for value in returns:
        wealth *= 1.0 + value
    return wealth - 1.0


def _rolling_horizon_returns(observations: Sequence[MarketObservation], horizon: int) -> List[float]:
    if len(observations) < horizon:
        return [item.simple_return for item in observations if item.simple_return]
    return [
        _compound_return([item.simple_return for item in observations[index - horizon + 1 : index + 1]])
        for index in range(horizon - 1, len(observations))
    ]


def run_backtest(
    observations: Iterable[MarketObservation],
    config: StudyConfig,
    strategies: Tuple[str, ...] = DEFAULT_STRATEGIES,
) -> Dict[str, List[TradeRecord]]:
    obs = list(observations)
    if not obs:
        return {name: [] for name in strategies}

    if any(not item.regime for item in obs):
        attach_regimes(obs)

    horizon = config.backtest.rebalance_every
    config.simulation.horizon_days = horizon

    weights = {name: 0.50 for name in strategies}
    results: Dict[str, List[TradeRecord]] = {name: [] for name in strategies}
    scenario_engine = RegimeAwareHestonEngine(config.simulation, config.regime_priors)

    start = max(config.backtest.train_window, config.forecaster.lookback)
    stop = len(obs) - horizon
    for current_index in range(start, stop, horizon):
        train_slice = obs[current_index - config.backtest.train_window : current_index + 1]
        history = train_slice[-config.forecaster.lookback :]

        forecaster = EnsembleVolatilityForecaster(config.forecaster, seed=current_index + config.forecaster.seed)
        forecaster.fit(train_slice)
        forecast = forecaster.predict(history)

        historical_horizon_returns = _rolling_horizon_returns(train_slice, horizon)
        static_heston_bundle = scenario_engine.simulate_static_heston_returns(
            current_price=obs[current_index].close,
            realized_vol=obs[current_index].realized_vol,
            regime=obs[current_index].regime,
            seed=10_000 + current_index,
            horizon_days=horizon,
        )
        lstm_only_bundle = scenario_engine.simulate_lstm_only_returns(
            forecast=forecast,
            regime=obs[current_index].regime,
            seed=20_000 + current_index,
            horizon_days=horizon,
        )
        hybrid_bundle = scenario_engine.simulate_hybrid_returns(
            current_price=obs[current_index].close,
            forecast=forecast,
            regime=obs[current_index].regime,
            seed=30_000 + current_index,
            horizon_days=horizon,
        )
        scenario_catalog = {
            "historical": {"returns": historical_horizon_returns, "summary": summarize_returns(historical_horizon_returns)},
            "static_heston": {"returns": static_heston_bundle.returns, "summary": static_heston_bundle.summary},
            "lstm_only": {"returns": lstm_only_bundle.returns, "summary": lstm_only_bundle.summary},
            "hybrid": {"returns": hybrid_bundle.returns, "summary": hybrid_bundle.summary},
        }

        realized_risky_return = _compound_return(
            [item.simple_return for item in obs[current_index + 1 : current_index + 1 + horizon]]
        )
        cash_return = config.optimizer.cash_rate * (horizon / 252.0)
        transaction_cost = config.optimizer.turnover_cost_bps / 10000.0

        for strategy in strategies:
            spec = STRATEGY_SPECS[strategy]
            scenario_source = spec["scenario_source"]
            scenario_returns = scenario_catalog[scenario_source]["returns"]
            scenario_summary = scenario_catalog[scenario_source]["summary"]

            if spec["objective_mode"] == "mean_variance":
                decision = optimize_mean_variance_allocation(
                    strategy=strategy,
                    scenario_source=scenario_source,
                    historical_returns=historical_horizon_returns,
                    previous_weight=weights[strategy],
                    config=config.optimizer,
                    horizon_days=horizon,
                )
            elif spec["objective_mode"] == "empirical_cvar":
                decision = optimize_mean_cvar_allocation(
                    strategy=strategy,
                    scenario_source=scenario_source,
                    scenario_returns=scenario_returns,
                    previous_weight=weights[strategy],
                    config=config.optimizer,
                    horizon_days=horizon,
                    epsilon=0.0,
                    robust=False,
                )
            else:
                epsilon = (
                    config.optimizer.fixed_epsilon
                    if spec.get("epsilon_mode") == "fixed"
                    else adaptive_epsilon(forecast.uncertainty, obs[current_index].regime, config.optimizer)
                )
                decision = optimize_mean_cvar_allocation(
                    strategy=strategy,
                    scenario_source=scenario_source,
                    scenario_returns=scenario_returns,
                    previous_weight=weights[strategy],
                    config=config.optimizer,
                    horizon_days=horizon,
                    epsilon=epsilon,
                    robust=True,
                )

            turnover = abs(decision.weight - weights[strategy])
            realized_portfolio_return = (
                decision.weight * realized_risky_return
                + (1.0 - decision.weight) * cash_return
                - transaction_cost * turnover
            )
            results[strategy].append(
                TradeRecord(
                    date=obs[current_index].date.isoformat(),
                    strategy=strategy,
                    scenario_source=scenario_source,
                    objective_mode=decision.objective_mode,
                    regime=obs[current_index].regime,
                    risky_weight=decision.weight,
                    cash_weight=round(1.0 - decision.weight, 6),
                    epsilon=decision.epsilon,
                    eta=decision.eta,
                    forecast_vol=forecast.predicted_vol,
                    uncertainty=forecast.uncertainty,
                    expected_return=decision.expected_return,
                    empirical_cvar_loss=decision.cvar,
                    robust_cvar_loss=decision.robust_cvar,
                    realized_return=realized_portfolio_return,
                    turnover=turnover,
                    scenario_mean=scenario_summary.get("mean", 0.0),
                    scenario_q05=scenario_summary.get("q05", 0.0),
                    training_rmse_log_vol=forecast.diagnostics.get("training", {}).get("rmse_log_vol", 0.0),
                )
            )
            weights[strategy] = decision.weight

    return results
