# DRO Gap Closure

## Gap 1: Formal DRO Optimization Layer

The project now solves an explicit Wasserstein distributionally robust mean-CVaR problem for an index-plus-cash portfolio.

- Portfolio variable: `w in [0, 1]`
- Loss function: `L(w, r) = -(w * r + (1 - w) * r_f)`
- Empirical distribution: `P_hat_N = (1/N) sum_i delta_{r_i}`
- Ambiguity set: `P_epsilon = { Q : W_1(Q, P_hat_N) <= epsilon }`

The code uses the specialized dual:

`min_(w, eta) -mu_hat(w) + gamma * [ eta + ( mean((L_i(w)-eta)_+) + epsilon * |w| ) / (1-alpha) ]`

This is implemented in `dro_study/dro.py`.

## Gap 2: Baseline Comparison

The pipeline now compares:

- Mean-Variance
- Vanilla CVaR
- Static Heston (no LSTM)
- LSTM Only (no Heston)
- LSTM + Heston (no DRO)
- Fixed-epsilon DRO
- Adaptive LSTM + Heston + DRO

## Gap 3: Backtesting

The pipeline now includes:

- Rolling estimation window
- Rebalancing
- Transaction costs
- Out-of-sample realized performance tracking
- Regime-sliced reporting

This is implemented in `dro_study/backtest.py`.

## Gap 4: Problem Statement

The project statement is now explicit:

Traditional optimizers fail when return distributions shift across macro regimes.  
Pure Heston misses adaptive state estimation.  
Pure LSTM misses stochastic-volatility structure.  
The research question is whether combining LSTM state estimation, Heston scenario structure, and Wasserstein-DRO improves tail protection under regime shifts.

## Gap 5: Ablation Study

The ablations now isolate:

- Only Heston
- Only LSTM
- No DRO

This is reported automatically in `report.json` and `research_summary.md`.

## Gap 6: Interpretation Layer

The generated summary now translates the output into financial implications:

- Crash protection
- Tail-risk reduction
- Regime awareness
- Benefit of adaptive versus fixed robustness

This is generated in `dro_study/reporting.py`.
