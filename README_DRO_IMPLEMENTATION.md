# DRO Study Implementation

This codebase now closes the six gaps from the project review:

- A formal Wasserstein distributionally robust mean-CVaR layer is implemented instead of only measuring CVaR after simulation.
- The backtest now compares `Mean-Variance`, `Vanilla CVaR`, `Static Heston`, `LSTM Only`, `LSTM + Heston (No DRO)`, `Fixed DRO`, and `Adaptive DRO`.
- A rolling out-of-sample backtest with transaction costs and rebalancing is built into the main pipeline.
- The generated report includes a sharper problem statement and research question.
- The generated outputs include explicit ablation comparisons for `only Heston`, `only LSTM`, and `no DRO`.
- A research summary file now turns the results into financial implications such as crash protection, tail estimation, and regime awareness.

## Files

- `run_dro_study.py`: command-line entry point.
- `dro_study/data.py`: CSV loading, synthetic demo data, and engineered features.
- `dro_study/regimes.py`: regime inference and one-hot encoding.
- `dro_study/forecasting.py`: pure-Python neural forecaster plus constrained Heston parameter head.
- `dro_study/heston.py`: static Heston, LSTM-only, and hybrid scenario generators.
- `dro_study/dro.py`: empirical mean-CVaR and Wasserstein-DRO optimization.
- `dro_study/backtest.py`: rolling evaluation engine and strategy catalog.
- `dro_study/reporting.py`: metrics, ablations, financial-implication summary, and markdown report writer.

## Strategy Set

- `mean_variance`: Markowitz benchmark.
- `vanilla_cvar`: historical empirical CVaR benchmark.
- `static_heston_cvar`: only Heston.
- `lstm_only_cvar`: only LSTM.
- `hybrid_cvar`: LSTM + Heston, no DRO.
- `fixed_dro`: hybrid model with fixed epsilon.
- `adaptive_dro`: hybrid model with adaptive epsilon.

## Run

Synthetic demo:

```bash
python3 run_dro_study.py
```

With your own CSV:

```bash
python3 run_dro_study.py --input-csv your_data.csv
```

Expected CSV columns:

- `date`
- `close`
- `vix` or `vvix`
- `cpi`
- `fed_funds`

## Outputs

Outputs are written into `dro_outputs/` by default:

- `report.json`
- `research_summary.md`
- `trades_mean_variance.csv`
- `trades_vanilla_cvar.csv`
- `trades_static_heston_cvar.csv`
- `trades_lstm_only_cvar.csv`
- `trades_hybrid_cvar.csv`
- `trades_fixed_dro.csv`
- `trades_adaptive_dro.csv`

## Notes

- The implementation is standard-library-only so it runs in this folder without external installs.
- The formal DRO layer is specialized to a tractable index-plus-cash setting, which is consistent with the current project framing and gives an explicit optimization problem rather than a descriptive tail-risk plot.
