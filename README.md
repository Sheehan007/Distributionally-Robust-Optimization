# Distributionally Robust Optimization Under Stochastic Volatility

## Overview

This repository implements a research-oriented pipeline for studying **distributionally robust portfolio optimization under stochastic volatility and macro regime shifts**. The goal is to move beyond a simple simulation or a point-prediction exercise and build a complete workflow that:

1. estimates market state using a neural volatility forecaster,
2. imposes stochastic-volatility structure through a Heston scenario engine,
3. solves an explicit Wasserstein distributionally robust mean-CVaR portfolio problem,
4. benchmarks the resulting strategy against meaningful baselines and ablations,
5. evaluates the portfolio through rolling out-of-sample backtesting.

The code is deliberately modular so that each modeling choice can be isolated, explained, and defended in an academic setting.

## Research Motivation

Classical portfolio models often fail when market conditions change abruptly. Mean-variance optimization assumes relatively stable second moments, while purely historical CVaR optimization assumes that the recent empirical distribution is informative enough for the next period. In practice, volatility clusters, crash risk is asymmetric, and macro regimes matter.

This project addresses that problem by combining:

- **LSTM-style state estimation** for adaptive volatility forecasting,
- **Heston stochastic-volatility structure** for coherent scenario generation,
- **Wasserstein DRO** for robust portfolio choice under distributional uncertainty.

The central research question is:

> Can a portfolio that combines neural state estimation, Heston structure, and Wasserstein-distributionally robust mean-CVaR optimization deliver stronger tail protection under regime shifts than simpler baselines?

## The Story of the Project

This repository is easiest to understand as a story rather than as a stack of isolated files.

The project started with a straightforward but important observation: before making a robust portfolio decision, we first need a believable model of how market uncertainty behaves. That is why the early part of the work focused on the **Heston stochastic-volatility model**. Instead of treating volatility as fixed, the model allows it to evolve through time:

`dS_t = mu_t S_t dt + sqrt(v_t) S_t dW_t^S`

`dv_t = kappa_t (theta_t - v_t) dt + sigma_{v,t} sqrt(v_t) dW_t^v`

with correlation between the price shock and the volatility shock. In practical terms, this lets the model express ideas that traders and risk managers actually care about: **volatility clustering**, **mean reversion in variance**, and the **leverage effect** where falling prices are associated with rising volatility.

Once the Heston simulations were plotted, the first financial intuition became clear. The asset-price paths were not smooth, evenly spaced geometric Brownian-motion paths. Some widened dramatically, some stayed calm, and the variance process spiked in bursts. That visual behavior mattered because it showed that the model could represent stress periods instead of pretending every day looks average. The density plots pushed the story further: under negative price-volatility correlation, the left tail became meaningfully heavier. In other words, the model naturally produced the kind of crash asymmetry that motivates distributionally robust optimization in the first place.

The implied-volatility smile reinforced that message. Higher implied volatility at lower strikes means the market prices downside risk more aggressively than upside risk. In the presentation, that became an important turning point: the project was no longer just about forecasting a return, but about understanding why tail-risk-aware decisions are necessary in equity markets.

The next chapter of the story was a failed idea, and that failure was useful. A direct LSTM was trained to predict next-day returns from recent returns and volatility. The result was not impressive: the model mostly converged to the mean. That was not just a machine-learning inconvenience; it was a conceptual lesson. Next-day returns are noisy and weakly predictable, so using a neural network as a point-return forecaster was not the right role for it.

That failure changed the direction of the project. Instead of asking the neural network to predict returns directly, the project reframed the network as a **state estimator for next-period volatility**. This shift is central to the current repository. The LSTM is now used to track volatility regimes, not to pretend it can perfectly forecast tomorrow’s return. In the presentation, the training curves and the actual-versus-predicted volatility plots told a much more convincing story: the network tracked major volatility shifts, spikes, and calm periods with much more credibility than the return-prediction setup.

At that point the project became a hybrid system. The neural model supplied an adaptive estimate of the current volatility state, while the Heston layer supplied structural discipline and realistic forward scenarios. This is where the **Feller condition** became part of the narrative rather than a purely technical side note. The model should not produce nonsensical variance dynamics, so the parameterization is constrained to remain theoretically consistent with stochastic-volatility theory:

`2 kappa_t theta_t > sigma_{v,t}^2`

That design choice matters academically because it shows that the machine-learning layer is being used to support the mathematical model, not replace it with an unconstrained black box.

Once the LSTM and Heston components were coupled, the scenario outputs became noticeably more informative. In the presentation, the hybrid tail-risk plots were more dynamic than static Heston. During stressed periods, the hybrid engine produced deeper negative CVaR and sharper movements in the scenario distribution. That was exactly the behavior the project needed: a regime-aware engine that reacts when the market environment changes instead of leaving the scenario distribution frozen.

Only after reaching that point does the final chapter make sense: **distributionally robust portfolio optimization**. The project does not stop at generating scenarios and measuring CVaR. It asks a harder question: if the predictive distribution itself may be wrong, what portfolio should be chosen under a neighborhood of plausible distributions? That is why the optimization layer uses a Wasserstein ambiguity set and solves a robust mean-CVaR problem instead of trusting the estimated distribution literally.

So the story of the repository is a progression:

1. start with stochastic-volatility intuition,
2. learn why direct return prediction is too weak,
3. re-purpose the neural model as a volatility state estimator,
4. combine that state estimate with Heston structure to generate better scenarios,
5. use DRO to make the final portfolio decision robust to distributional misspecification.

This progression is important because it explains *why* each file exists. The repository is not just a set of modules; it is the record of how the project matured from a simulation exercise into a defensible research workflow.

## What This Repository Now Covers

The repository closes the main methodological gaps that would otherwise make the project look incomplete:

- a **formal DRO optimization layer**, not just CVaR measurement after simulation,
- **baseline comparisons** against classical and non-robust alternatives,
- a **rolling backtest** instead of only model-validation plots,
- a sharper **problem statement and research framing**,
- **ablation studies** to isolate which part of the architecture adds value,
- an **interpretation layer** translating outputs into financial implications.

## Repository Structure

### Top-level Files

- `heston2.py`  
  This is the original monolithic prototype already present in the repository. It combines data loading, LSTM training, and Heston-style scenario generation in a single script using the scientific Python stack. It is valuable as the project’s earlier experimental baseline, while the newer `dro_study/` package is the cleaner research-oriented and modular rewrite.

- `run_dro_study.py`  
  This is the main command-line entry point. It loads either a user-supplied dataset or a synthetic demo dataset, runs the full pipeline, and writes the outputs. If someone wants to reproduce the project quickly, this is the first file they should run.

- `README.md`  
  This file is the high-level narrative guide for the repository. It explains the research motivation, the pipeline design, the file structure, and how the code should be interpreted by professors, peers, or reviewers.

- `README_DRO_IMPLEMENTATION.md`  
  This is a shorter implementation note focused on how the six project gaps were closed in code. It is useful for a fast technical overview.

- `DRO_GAP_CLOSURE.md`  
  This file is a concise methodological note that explicitly maps the codebase to the identified gaps: formal DRO, baselines, backtesting, problem statement, ablations, and interpretation.

- `.gitignore`  
  This keeps transient files such as `__pycache__`, smoke-test outputs, and macOS metadata out of version control so the repository stays clean.

### Package: `dro_study/`

- `dro_study/__init__.py`  
  This exposes the key package entry points and makes the project importable as a cohesive toolkit instead of a loose collection of scripts.

- `dro_study/config.py`  
  This file stores the configuration dataclasses for the study. It contains the forecaster settings, simulation settings, optimizer settings, backtest settings, and regime priors. In a research workflow, this file is important because it centralizes assumptions and makes experiments reproducible.

- `dro_study/data.py`  
  This file handles data ingestion and preprocessing. It defines the `MarketObservation` structure, loads market data from CSV, computes log returns and realized volatility, and can generate a synthetic regime-aware dataset for smoke testing. This ensures that the rest of the pipeline receives clean, standardized inputs.

- `dro_study/regimes.py`  
  This module turns inflation and policy-rate changes into explicit macro regime labels. It is where the project becomes regime-aware rather than purely time-series driven. The regimes are then used both in scenario generation and in the adaptive robustness layer.

- `dro_study/forecasting.py`  
  This module implements the neural volatility/state-estimation layer. It predicts next-period volatility, estimates uncertainty, and transforms those predictions into Heston-compatible parameters while enforcing structural constraints such as positivity and the Feller condition. Conceptually, this is where machine learning supports, rather than replaces, the financial model.

- `dro_study/heston.py`  
  This file contains the scenario generators. It supports:
  - a **static Heston** baseline with no neural update,
  - an **LSTM-only** distributional baseline with no Heston structure,
  - a **hybrid LSTM + Heston** model that injects state estimates into stochastic-volatility simulation.
  
  This file is crucial because it makes the ablation study possible and ensures that the project is not just a single black-box model.

- `dro_study/dro.py`  
  This is the core optimization layer. It implements:
  - mean-variance allocation,
  - empirical mean-CVaR allocation,
  - a formal **Wasserstein distributionally robust mean-CVaR** objective.
  
  It includes the specialized dual-style reformulation used in the one-risky-asset-plus-cash setting. In terms of research contribution, this is one of the most important files in the repository.

- `dro_study/backtest.py`  
  This file coordinates the rolling out-of-sample backtest. It repeatedly refits the forecaster, generates scenarios, solves the relevant portfolio problem, applies transaction costs, and records realized portfolio returns. It is the bridge between model construction and actual strategy evaluation.

- `dro_study/reporting.py`  
  This file converts raw trade records into a research-ready report. It computes performance metrics, regime-sliced summaries, ablation deltas, financial implications, and a markdown research summary. This is the interpretation layer that helps explain what the models are doing and why the differences matter.

## Strategy Set

The repository compares several strategy classes so that the final result can be defended as research rather than a one-off simulation:

- `mean_variance`  
  Classical Markowitz-style benchmark using rolling historical returns.

- `vanilla_cvar`  
  Historical empirical mean-CVaR benchmark with no stochastic-volatility model and no DRO layer.

- `static_heston_cvar`  
  Heston-only baseline using regime priors but no neural state update.

- `lstm_only_cvar`  
  Neural-volatility-only baseline with no Heston structure.

- `hybrid_cvar`  
  LSTM + Heston hybrid with no DRO. This isolates the value of the robust decision layer.

- `fixed_dro`  
  Hybrid model with a constant Wasserstein radius.

- `adaptive_dro`  
  Hybrid model with regime- and uncertainty-adaptive robustness. This is the main target model.

## Formal Optimization View

The risky-asset weight is denoted by `w in [0,1]`, with the remaining capital allocated to cash. Let `r` denote the risky return and `r_f` the cash return. The portfolio loss is modeled as:

`L(w, r) = -(w * r + (1 - w) * r_f)`

The empirical scenario distribution is:

`P_hat_N = (1/N) sum_i delta_{r_i}`

The Wasserstein ambiguity set is:

`P_epsilon = { Q : W_1(Q, P_hat_N) <= epsilon }`

The code solves a specialized one-risky-asset mean-CVaR DRO problem of the form:

`maximize_w  E_{P_hat_N}[w r + (1-w) r_f] - gamma * sup_{Q in P_epsilon} CVaR_alpha^Q(L(w, r))`

using the explicit finite-dimensional reformulation implemented in `dro_study/dro.py`.

## How To Run

### Synthetic smoke test

```bash
python3 run_dro_study.py
```

### With your own dataset

```bash
python3 run_dro_study.py --input-csv your_data.csv
```

### Expected CSV columns

- `date`
- `close`
- `vix` or `vvix`
- `cpi`
- `fed_funds`

## Output Files

When the pipeline runs, it writes outputs such as:

- `report.json`  
  Structured report with performance metrics, ablation deltas, sensitivity results, and research framing.

- `research_summary.md`  
  Human-readable research summary including the problem statement, formal DRO layer, strategy comparison table, ablation table, and financial implications.

- `trades_*.csv`  
  Per-strategy trade records showing dates, weights, epsilon values, scenario statistics, and realized returns.

## Why This Is Useful Academically

For a professor or peer reviewer, the value of this repository is that it makes the modeling logic transparent. Each layer of the project has a separate module and a separate purpose:

- the **data layer** prepares features and regimes,
- the **forecasting layer** estimates state,
- the **scenario layer** creates coherent return distributions,
- the **optimization layer** makes the portfolio decision,
- the **backtest layer** evaluates the decision,
- the **reporting layer** explains the result.

That structure makes the project easier to critique, defend, and extend.

## Conceptual References From the Presentation

The presentation that motivated this repository draws on several strands of literature:

- Wasserstein distributionally robust optimization, especially work by **Esfahani and Kuhn**.
- Broader DRO framing and interpretation from **Rahimian and Mehrotra**.
- Hybrid LSTM-based stock-market modeling ideas from **Botunac, Bosna, and Matetic**.
- Regime thinking informed by work on **clustering market regimes with the Wasserstein distance**.

These references matter because the repository is trying to connect three traditions at once: stochastic calculus, robust optimization, and data-driven state estimation.

## Notes

- The code is **standard-library-only** so it can run even in restricted environments without extra package installation.
- The DRO implementation is specialized to a **tractable index-plus-cash setting**, which keeps the optimization transparent and computationally lightweight.
- The synthetic smoke-test outputs are useful for validation of the software pipeline, but the most meaningful results will come from real SPY/VIX/CPI/Fed Funds data.
