"""Reporting, ablation analysis, and human-readable research summaries."""

from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from .backtest import DEFAULT_STRATEGIES, TradeRecord, run_backtest
from .config import StudyConfig
from .data import MarketObservation
from .dro import build_formal_dro_formulation
from .regimes import REGIME_LABELS


STRATEGY_LABELS = {
    "mean_variance": "Mean-Variance",
    "vanilla_cvar": "Vanilla CVaR",
    "static_heston_cvar": "Static Heston (No LSTM)",
    "lstm_only_cvar": "LSTM Only (No Heston)",
    "hybrid_cvar": "LSTM + Heston (No DRO)",
    "fixed_dro": "Fixed-epsilon DRO",
    "adaptive_dro": "Adaptive LSTM + Heston + DRO",
}


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return mean(values) if values else default


def _tail_mean(returns: List[float], alpha: float = 0.95) -> float:
    if not returns:
        return 0.0
    tail_count = max(1, int(math.ceil(len(returns) * (1.0 - alpha))))
    return _safe_mean(sorted(returns)[:tail_count], 0.0)


def _max_drawdown(returns: List[float]) -> float:
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        drawdown = (wealth / peak) - 1.0
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _format_bps(value: float) -> str:
    return f"{10_000.0 * value:.1f} bps"


def compute_metrics(records: List[TradeRecord], periods_per_year: float) -> Dict[str, float]:
    returns = [item.realized_return for item in records]
    turnovers = [item.turnover for item in records]
    epsilons = [item.epsilon for item in records]
    weights = [item.risky_weight for item in records]
    if not returns:
        return {
            "periods": 0,
            "cagr": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "tail_cvar_95": 0.0,
            "max_drawdown": 0.0,
            "average_turnover": 0.0,
            "average_epsilon": 0.0,
            "average_risky_weight": 0.0,
        }

    avg_return = _safe_mean(returns)
    vol = pstdev(returns) if len(returns) > 1 else 0.0
    annualized_vol = vol * math.sqrt(periods_per_year)
    sharpe = (avg_return / vol) * math.sqrt(periods_per_year) if vol > 1e-12 else 0.0

    wealth = 1.0
    for value in returns:
        wealth *= 1.0 + value
    years = len(returns) / periods_per_year if periods_per_year else 0.0
    cagr = wealth ** (1.0 / years) - 1.0 if years > 0 and wealth > 0 else wealth - 1.0

    return {
        "periods": len(returns),
        "cagr": cagr,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "tail_cvar_95": _tail_mean(returns, 0.95),
        "max_drawdown": _max_drawdown(returns),
        "average_turnover": _safe_mean(turnovers, 0.0),
        "average_epsilon": _safe_mean(epsilons, 0.0),
        "average_risky_weight": _safe_mean(weights, 0.0),
    }


def compute_regime_breakdown(records: List[TradeRecord], periods_per_year: float) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[TradeRecord]] = {}
    for record in records:
        grouped.setdefault(record.regime, []).append(record)
    return {
        REGIME_LABELS.get(regime, regime): compute_metrics(items, periods_per_year)
        for regime, items in grouped.items()
    }


def build_problem_statement() -> Dict[str, str]:
    return {
        "motivation": (
            "Traditional mean-variance and historical CVaR portfolios fail when volatility regimes shift "
            "because they treat the return distribution as too stable and too well estimated."
        ),
        "research_gap": (
            "Purely statistical forecasts miss stochastic-volatility structure, while pure Heston models miss "
            "time-varying state information and uncertainty-aware decision rules."
        ),
        "project_question": (
            "Can a portfolio that combines LSTM state estimation, Heston scenario structure, and Wasserstein "
            "distributionally robust mean-CVaR optimization deliver better tail protection under macro regime shifts?"
        ),
        "hypothesis": (
            "The hybrid model should improve crash protection and regime awareness, and the adaptive DRO layer "
            "should further reduce tail losses relative to the same hybrid model without robustness."
        ),
    }


def build_strategy_catalog() -> Dict[str, str]:
    return {
        "mean_variance": "Markowitz baseline using historical rolling returns only.",
        "vanilla_cvar": "Empirical mean-CVaR optimizer using historical rolling return scenarios only.",
        "static_heston_cvar": "Regime-dependent Heston simulator without any neural volatility update.",
        "lstm_only_cvar": "Neural volatility forecaster with no stochastic-volatility structure.",
        "hybrid_cvar": "LSTM state estimation plus Heston scenario generation, but no DRO ambiguity set.",
        "fixed_dro": "Hybrid scenario engine plus a constant Wasserstein radius.",
        "adaptive_dro": "Hybrid scenario engine plus a regime- and uncertainty-adaptive Wasserstein radius.",
    }


def build_ablation_summary(metrics: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    anchor = "adaptive_dro"
    if anchor not in metrics:
        return {"anchor": anchor, "comparisons": {}}

    anchor_metrics = metrics[anchor]
    comparisons = {}
    baseline_groups = {
        "mean_variance": "Classical benchmark",
        "vanilla_cvar": "Historical-tail benchmark",
        "static_heston_cvar": "Only Heston ablation",
        "lstm_only_cvar": "Only LSTM ablation",
        "hybrid_cvar": "No DRO ablation",
        "fixed_dro": "Fixed-radius robustness benchmark",
    }

    for strategy, label in baseline_groups.items():
        if strategy not in metrics:
            continue
        base = metrics[strategy]
        comparisons[strategy] = {
            "label": label,
            "sharpe_delta": anchor_metrics["sharpe"] - base["sharpe"],
            "tail_cvar_delta": anchor_metrics["tail_cvar_95"] - base["tail_cvar_95"],
            "max_drawdown_delta": anchor_metrics["max_drawdown"] - base["max_drawdown"],
            "cagr_delta": anchor_metrics["cagr"] - base["cagr"],
        }
    return {"anchor": anchor, "comparisons": comparisons}


def build_financial_implications(metrics: Dict[str, Dict[str, float]], ablation: Dict[str, object]) -> List[str]:
    implications: List[str] = []
    anchor = ablation.get("anchor", "adaptive_dro")
    comparisons = ablation.get("comparisons", {})
    if anchor not in metrics:
        return implications

    def narrative(
        label: str,
        sharpe_delta: float,
        cvar_delta: float,
        drawdown_delta: float,
        cagr_delta: float,
    ) -> str:
        gains = []
        if sharpe_delta > 1e-8:
            gains.append(f"Sharpe by {sharpe_delta:.3f}")
        if cvar_delta > 1e-8:
            gains.append(f"tail CVaR by {_format_bps(cvar_delta)}")
        if drawdown_delta > 1e-8:
            gains.append(f"max drawdown by {_format_pct(drawdown_delta)}")
        if cagr_delta > 1e-8:
            gains.append(f"CAGR by {_format_pct(cagr_delta)}")
        if gains:
            return f"Against {label}, adaptive DRO improves " + ", ".join(gains) + "."
        return (
            f"Against {label}, the current run does not show a clear improvement from adaptive DRO; "
            "this means the architecture is in place, but the benefit still depends on the data and calibration."
        )

    if "mean_variance" in comparisons:
        delta = comparisons["mean_variance"]
        implications.append(
            narrative(
                "classical mean-variance",
                delta["sharpe_delta"],
                delta["tail_cvar_delta"],
                delta["max_drawdown_delta"],
                delta["cagr_delta"],
            )
        )
    if "static_heston_cvar" in comparisons:
        delta = comparisons["static_heston_cvar"]
        implications.append(
            narrative(
                "static Heston, isolating the value of neural state estimation",
                delta["sharpe_delta"],
                delta["tail_cvar_delta"],
                delta["max_drawdown_delta"],
                delta["cagr_delta"],
            )
        )
    if "lstm_only_cvar" in comparisons:
        delta = comparisons["lstm_only_cvar"]
        implications.append(
            narrative(
                "LSTM-only forecasting, isolating the value of Heston structure",
                delta["sharpe_delta"],
                delta["tail_cvar_delta"],
                delta["max_drawdown_delta"],
                delta["cagr_delta"],
            )
        )
    if "hybrid_cvar" in comparisons:
        delta = comparisons["hybrid_cvar"]
        implications.append(
            narrative(
                "the hybrid no-DRO model, isolating the value of the robust decision layer",
                delta["sharpe_delta"],
                delta["tail_cvar_delta"],
                delta["max_drawdown_delta"],
                delta["cagr_delta"],
            )
        )
    if "fixed_dro" in comparisons:
        delta = comparisons["fixed_dro"]
        implications.append(
            narrative(
                "fixed-epsilon DRO, isolating the value of adaptive robustness",
                delta["sharpe_delta"],
                delta["tail_cvar_delta"],
                delta["max_drawdown_delta"],
                delta["cagr_delta"],
            )
        )
    return implications


def run_sensitivity(observations: List[MarketObservation], base_config: StudyConfig) -> List[Dict[str, float]]:
    epsilon_scales = (0.75, 1.00, 1.25)
    turnover_costs = (base_config.optimizer.turnover_cost_bps, base_config.optimizer.turnover_cost_bps * 2.0)
    rebalance_steps = (base_config.backtest.rebalance_every, base_config.backtest.rebalance_every * 2)

    sensitivity_rows: List[Dict[str, float]] = []
    for epsilon_scale in epsilon_scales:
        for turnover_cost in turnover_costs:
            for rebalance_every in rebalance_steps:
                config = deepcopy(base_config)
                config.optimizer.uncertainty_scale *= epsilon_scale
                config.optimizer.turnover_cost_bps = turnover_cost
                config.backtest.rebalance_every = rebalance_every
                config.simulation.horizon_days = rebalance_every

                records = run_backtest(observations, config, strategies=("adaptive_dro",))["adaptive_dro"]
                metrics = compute_metrics(records, periods_per_year=252.0 / rebalance_every)
                sensitivity_rows.append(
                    {
                        "epsilon_scale": epsilon_scale,
                        "turnover_cost_bps": turnover_cost,
                        "rebalance_every": rebalance_every,
                        "cagr": metrics["cagr"],
                        "sharpe": metrics["sharpe"],
                        "tail_cvar_95": metrics["tail_cvar_95"],
                        "max_drawdown": metrics["max_drawdown"],
                    }
                )
    return sensitivity_rows


def generate_report(
    observations: List[MarketObservation],
    strategy_records: Dict[str, List[TradeRecord]],
    config: StudyConfig,
    include_sensitivity: bool = True,
) -> Dict[str, object]:
    periods_per_year = 252.0 / config.backtest.rebalance_every
    metrics = {
        strategy: compute_metrics(records, periods_per_year)
        for strategy, records in strategy_records.items()
    }
    regime_breakdown = {
        strategy: compute_regime_breakdown(records, periods_per_year)
        for strategy, records in strategy_records.items()
    }
    ablation = build_ablation_summary(metrics)
    financial_implications = build_financial_implications(metrics, ablation)

    best_sharpe_strategy = max(metrics, key=lambda key: metrics[key]["sharpe"]) if metrics else ""
    smallest_drawdown_strategy = max(metrics, key=lambda key: metrics[key]["max_drawdown"]) if metrics else ""

    return {
        "headline": {
            "best_sharpe_strategy": best_sharpe_strategy,
            "best_sharpe": metrics.get(best_sharpe_strategy, {}).get("sharpe", 0.0),
            "least_severe_drawdown_strategy": smallest_drawdown_strategy,
            "least_severe_drawdown": metrics.get(smallest_drawdown_strategy, {}).get("max_drawdown", 0.0),
            "observation_count": len(observations),
        },
        "problem_statement": build_problem_statement(),
        "formal_dro_formulation": build_formal_dro_formulation(config.optimizer.alpha),
        "strategy_catalog": build_strategy_catalog(),
        "gap_coverage": {
            "gap_1_formal_dro_optimization": "Closed via an explicit Wasserstein mean-CVaR formulation and specialized dual in the optimizer.",
            "gap_2_baseline_comparison": "Closed via mean-variance, vanilla CVaR, static Heston, hybrid no-DRO, fixed DRO, and adaptive DRO comparisons.",
            "gap_3_backtesting": "Closed via a rolling-window backtest with rebalancing, transaction costs, and out-of-sample realized returns.",
            "gap_4_problem_statement": "Closed via an explicit research motivation, question, and hypothesis section in the generated report.",
            "gap_5_ablation": "Closed via only-Heston, only-LSTM, and no-DRO ablation strategies.",
            "gap_6_interpretation": "Closed via generated financial implications tied to crash protection, tail estimation, and regime awareness.",
        },
        "metrics": metrics,
        "regime_breakdown": regime_breakdown,
        "ablation": ablation,
        "financial_implications": financial_implications,
        "sensitivity": run_sensitivity(observations, config) if include_sensitivity else [],
    }


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header_line, divider, body]) if body else "\n".join([header_line, divider])


def build_markdown_summary(report: Dict[str, object]) -> str:
    problem = report["problem_statement"]
    formulation = report["formal_dro_formulation"]
    metrics = report["metrics"]
    ablation = report["ablation"]["comparisons"]

    metric_rows = []
    for strategy in DEFAULT_STRATEGIES:
        if strategy not in metrics:
            continue
        item = metrics[strategy]
        metric_rows.append(
            [
                STRATEGY_LABELS.get(strategy, strategy),
                _format_pct(item["cagr"]),
                f"{item['sharpe']:.3f}",
                _format_pct(item["tail_cvar_95"]),
                _format_pct(item["max_drawdown"]),
                _format_pct(item["average_turnover"]),
            ]
        )

    ablation_rows = []
    for strategy, delta in ablation.items():
        ablation_rows.append(
            [
                STRATEGY_LABELS.get(strategy, strategy),
                f"{delta['sharpe_delta']:.3f}",
                _format_bps(delta["tail_cvar_delta"]),
                _format_pct(delta["max_drawdown_delta"]),
                _format_pct(delta["cagr_delta"]),
            ]
        )

    lines = [
        "# DRO Study Research Summary",
        "",
        "## Problem Statement",
        f"- Motivation: {problem['motivation']}",
        f"- Research gap: {problem['research_gap']}",
        f"- Research question: {problem['project_question']}",
        f"- Hypothesis: {problem['hypothesis']}",
        "",
        "## Formal DRO Layer",
        f"- Ambiguity set: {formulation['ambiguity_set']}",
        f"- Loss: {formulation['loss_function']}",
        f"- Robust problem: {formulation['robust_problem']}",
        f"- Specialized dual used in code: {formulation['dual_reformulation_used_in_code']}",
        "",
        "## Strategy Comparison",
        _markdown_table(
            ["Strategy", "CAGR", "Sharpe", "Tail CVaR(95%)", "Max Drawdown", "Avg Turnover"],
            metric_rows,
        ),
        "",
        "## Ablation Against Adaptive DRO",
        _markdown_table(
            ["Baseline", "Sharpe Delta", "Tail CVaR Delta", "Max DD Delta", "CAGR Delta"],
            ablation_rows,
        ),
        "",
        "## Financial Implications",
    ]
    for line in report["financial_implications"]:
        lines.append(f"- {line}")

    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: str | Path,
    report: Dict[str, object],
    strategy_records: Dict[str, List[TradeRecord]],
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_path = destination / "research_summary.md"
    summary_path.write_text(build_markdown_summary(report), encoding="utf-8")

    for strategy, records in strategy_records.items():
        csv_path = destination / f"trades_{strategy}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(records[0].to_dict().keys()) if records else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                for record in records:
                    writer.writerow(record.to_dict())
