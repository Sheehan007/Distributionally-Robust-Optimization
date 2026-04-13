"""Formal empirical and Wasserstein-robust portfolio optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List

from .config import OptimizerConfig


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return mean(values) if values else default


def _variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _safe_mean(values)
    return _safe_mean([(value - avg) ** 2 for value in values], 0.0)


def portfolio_returns(weight: float, risky_returns: List[float], cash_return: float) -> List[float]:
    return [weight * value + (1.0 - weight) * cash_return for value in risky_returns]


def losses_from_returns(returns: List[float]) -> List[float]:
    return [-value for value in returns]


def empirical_cvar_from_losses(losses: List[float], alpha: float) -> tuple[float, float]:
    """
    Solve the Rockafellar-Uryasev empirical CVaR problem:

        CVaR_alpha(L) = min_eta eta + (1/(1-alpha)) * E[(L-eta)_+]
    """

    if not losses:
        return 0.0, 0.0

    eta_candidates = sorted(set(losses))
    lower = min(losses) - 1e-8
    upper = max(losses) + 1e-8

    best_eta = 0.0
    best_value = math.inf
    for eta in [lower, *eta_candidates, upper]:
        excess = _safe_mean([max(loss - eta, 0.0) for loss in losses], 0.0)
        value = eta + excess / max(1.0 - alpha, 1e-8)
        if value < best_value:
            best_value = value
            best_eta = eta

    return best_value, best_eta


def wasserstein_robust_cvar_from_losses(
    losses: List[float],
    alpha: float,
    epsilon: float,
    lipschitz_constant: float,
) -> tuple[float, float]:
    """
    Specialized finite-dimensional Wasserstein-DRO reformulation used in this codebase.

    For the scalar index-vs-cash setup with loss L(w, r) and a 1-Wasserstein ball
    around the empirical risky-return distribution, the robust CVaR is implemented as:

        min_eta eta + ( E[(L-eta)_+] + epsilon * Lip(L) ) / (1-alpha)

    where Lip(L) = |w| under the absolute-return ground metric.
    """

    if not losses:
        return 0.0, 0.0

    eta_candidates = sorted(set(losses))
    lower = min(losses) - 1e-8
    upper = max(losses) + 1e-8

    best_eta = 0.0
    best_value = math.inf
    penalty = epsilon * abs(lipschitz_constant)
    scale = max(1.0 - alpha, 1e-8)

    for eta in [lower, *eta_candidates, upper]:
        excess = _safe_mean([max(loss - eta, 0.0) for loss in losses], 0.0)
        value = eta + (excess + penalty) / scale
        if value < best_value:
            best_value = value
            best_eta = eta

    return best_value, best_eta


def cvar_loss(returns: List[float], alpha: float) -> float:
    losses = losses_from_returns(returns)
    value, _ = empirical_cvar_from_losses(losses, alpha)
    return value


def adaptive_epsilon(uncertainty: float, regime: str, config: OptimizerConfig) -> float:
    return config.epsilon_base + config.uncertainty_scale * uncertainty + config.regime_penalties.get(regime, 0.0)


def build_formal_dro_formulation(alpha: float) -> dict:
    return {
        "portfolio_set": "w in [0, 1], interpreted as the risky-asset weight in an index-plus-cash portfolio.",
        "empirical_distribution": "P_hat_N = (1/N) * sum_i delta_{r_i} built from scenario returns or historical rolling returns.",
        "ambiguity_set": "P_epsilon = { Q : W_1(Q, P_hat_N) <= epsilon } using the absolute-return ground metric.",
        "loss_function": "L(w, r) = -(w * r + (1 - w) * r_f).",
        "robust_problem": (
            "maximize_w  E_{P_hat_N}[w r + (1-w) r_f] - gamma * sup_{Q in P_epsilon} CVaR_alpha^Q(L(w, r))"
        ),
        "dual_reformulation_used_in_code": (
            f"min_(w, eta) -mu_hat(w) + gamma * [ eta + ( mean((L_i(w)-eta)_+) + epsilon * |w| ) / (1-{alpha:.2f}) ]"
        ),
        "implementation_note": (
            "The code solves the specialized one-risky-asset dual on a weight grid and eta grid, "
            "which gives an explicit formal DRO layer rather than a post-hoc CVaR diagnostic."
        ),
    }


@dataclass
class OptimizerDecision:
    strategy: str
    objective_mode: str
    scenario_source: str
    weight: float
    epsilon: float
    eta: float
    expected_return: float
    variance: float
    cvar: float
    robust_cvar: float
    score: float
    lipschitz_constant: float


def optimize_mean_variance_allocation(
    strategy: str,
    scenario_source: str,
    historical_returns: List[float],
    previous_weight: float,
    config: OptimizerConfig,
    horizon_days: int,
) -> OptimizerDecision:
    step = config.weight_step
    steps = int(round(config.max_weight / step))
    turnover_cost = config.turnover_cost_bps / 10000.0
    cash_return = config.cash_rate * (horizon_days / 252.0)

    best: OptimizerDecision | None = None
    for index in range(steps + 1):
        weight = min(config.max_weight, index * step)
        turnover_penalty = turnover_cost * abs(weight - previous_weight)
        sample_returns = portfolio_returns(weight, historical_returns, cash_return)
        expected_return = _safe_mean(sample_returns, cash_return)
        variance = _variance(sample_returns)
        cvar = cvar_loss(sample_returns, config.alpha)
        score = expected_return - config.variance_penalty * variance - turnover_penalty

        decision = OptimizerDecision(
            strategy=strategy,
            objective_mode="mean_variance",
            scenario_source=scenario_source,
            weight=round(weight, 6),
            epsilon=0.0,
            eta=0.0,
            expected_return=expected_return,
            variance=variance,
            cvar=cvar,
            robust_cvar=cvar,
            score=score,
            lipschitz_constant=abs(weight),
        )
        if best is None or decision.score > best.score:
            best = decision

    assert best is not None
    return best


def optimize_mean_cvar_allocation(
    strategy: str,
    scenario_source: str,
    scenario_returns: List[float],
    previous_weight: float,
    config: OptimizerConfig,
    horizon_days: int,
    epsilon: float = 0.0,
    robust: bool = False,
) -> OptimizerDecision:
    step = config.weight_step
    steps = int(round(config.max_weight / step))
    turnover_cost = config.turnover_cost_bps / 10000.0
    cash_return = config.cash_rate * (horizon_days / 252.0)
    objective_mode = "wasserstein_dro" if robust else "empirical_cvar"

    best: OptimizerDecision | None = None
    for index in range(steps + 1):
        weight = min(config.max_weight, index * step)
        turnover_penalty = turnover_cost * abs(weight - previous_weight)
        sample_returns = portfolio_returns(weight, scenario_returns, cash_return)
        losses = losses_from_returns(sample_returns)

        empirical_cvar, eta = empirical_cvar_from_losses(losses, config.alpha)
        robust_cvar = empirical_cvar
        robust_eta = eta
        if robust:
            robust_cvar, robust_eta = wasserstein_robust_cvar_from_losses(
                losses=losses,
                alpha=config.alpha,
                epsilon=epsilon,
                lipschitz_constant=weight,
            )

        expected_return = _safe_mean(sample_returns, cash_return)
        variance = _variance(sample_returns)
        risk_term = robust_cvar if robust else empirical_cvar
        score = expected_return - config.risk_aversion * risk_term - turnover_penalty

        decision = OptimizerDecision(
            strategy=strategy,
            objective_mode=objective_mode,
            scenario_source=scenario_source,
            weight=round(weight, 6),
            epsilon=epsilon if robust else 0.0,
            eta=robust_eta if robust else eta,
            expected_return=expected_return,
            variance=variance,
            cvar=empirical_cvar,
            robust_cvar=robust_cvar,
            score=score,
            lipschitz_constant=abs(weight),
        )
        if best is None or decision.score > best.score:
            best = decision

    assert best is not None
    return best
