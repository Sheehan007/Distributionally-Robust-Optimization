# DRO under stochastic volatility: computational study

Sheehan Mathur · Software release 1.0.0 · 31 October 2026

Companion to **Distributionally Robust Portfolio Choice under Stochastic
Volatility: A Review and Reproducible Multi-Asset Study**.

[Public repository](https://github.com/Sheehan007/Distributionally-Robust-Optimization)
· [Versioned release](https://github.com/Sheehan007/Distributionally-Robust-Optimization/releases/tag/v1.0.0)
· [Manuscript source](paper/main.tex)

This is a selective review and computational study of established methods,
with no claim of a new DRO algorithm. A preliminary single-asset analysis over
2019–2024 motivated the design; that period is exploratory. The 2025 comparison
is retrospective confirmation, not a prospective or publicly preregistered test.

The pipeline combines a persistence-anchored residual LSTM, annual physical
Heston estimation, and a convex four-ETF mean–CVaR allocation with quadratic
risk, Euclidean Wasserstein regularization, and turnover costs. SPY, IWM, TLT,
and GLD form the universe. A separate cash-enabled experiment distinguishes
composition from de-risking.

The primary matched nominal/fixed-DRO/adaptive-DRO comparison uses **20,000
scenarios under each of three independent seeds**. A broader model comparison
and sensitivity grids use 2,000. The likelihood search uses tighter tolerances,
larger budgets, multiple common-random-number seeds, and five independent
4,096-particle checking evaluations. Parameter tables use two significant
figures, which are not uncertainty intervals.

## What is public

- Executable scripts, economic and numerical settings, starting vectors, and
  the complete LaTeX source with generated figures and tables.
- 168 trained LSTM checkpoints and 28 annual calibration records.
- Aggregate forecast, likelihood, portfolio, bootstrap, Monte Carlo, numerical
  validation, and concentration summaries.
- Provider URLs, retrieval metadata, original file hashes, and five canonical
  hashes covering the scientific inputs actually used.

**Provider observations, adjusted-price CSVs, raw API responses, and daily
market/forecast/portfolio caches are excluded.** Review [DATA_ACCESS.md](DATA_ACCESS.md)
before obtaining local inputs. Exact historical numerical replication requires
appropriate access and matching canonical hashes. Provider revisions can make
a current-data rerun differ from the reported snapshot; the verifier identifies
that difference explicitly.

## Install

Use Python 3.12. The recorded run used Python 3.12.14 on macOS arm64.
Pinned scientific dependencies are in `requirements.txt`; the full run
environment is recorded in `environment.json`.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p tmp/numba tmp/mpl results data
export OPENBLAS_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/tmp/numba"
export MPLCONFIGDIR="$PWD/tmp/mpl"
```

Cross-platform linear algebra and filtering can produce small numerical
differences. Installing the software grants no rights to provider data.

## Audit without provider observations

The independent solver comparison uses synthetic scenarios:

```sh
.venv/bin/python test_tail_allocator.py
.venv/bin/python audit_public_release.py
.venv/bin/python paper_assets.py
```

Public summaries suffice to regenerate all numerical tables and the forecast
and parameter figures. The archived wealth and allocation figures are also
included. Regenerating those paths requires local inputs and the full workflow.

## Reproduce the reported computation

Obtain appropriately authorized inputs in the documented format, or, after
reviewing the provider terms, use the optional downloader:

```sh
.venv/bin/python download.py --acknowledge-terms
.venv/bin/python verify_snapshot.py
.venv/bin/python forecast.py --reuse-checkpoints
.venv/bin/python calibrate.py
.venv/bin/python portfolio.py
.venv/bin/python paired_accuracy.py
.venv/bin/python analyze.py
.venv/bin/python analyze_accuracy.py
.venv/bin/python extra_checks.py
.venv/bin/python paper_assets.py
.venv/bin/python test_study.py
.venv/bin/python test_tail_allocator.py
```

Checkpoint replay exactly reconstructed all four archived forecast panels in
the recorded environment. Calibration reuses the published annual fit JSONs
and rebuilds causal state paths. To refit the parameters from the same starting
vectors, run `refit_accuracy.py`. To train from scratch, use a separate working
copy without forecast CSV caches and run `forecast.py` without the reuse flag.
Do not change settings while retaining derived caches from a different run.

`validate_full_scenarios.py` performs eight additional full-epigraph comparisons
at four fixed schedule positions using 20,000 scenarios. `test_study.py` has nine
mathematical/timing tests, and `test_tail_allocator.py` adds an independent
cash/fractional-tail solver comparison. All ten pass in the recorded run.
Solver statuses and numerical tolerances are reported in the paper rather than
being treated as guarantees of exact-arithmetic optimization or generalization.

## Build the manuscript

```sh
cd paper
tectonic main.tex --keep-intermediates
```

A conventional LaTeX/BibTeX installation can also build it. The arXiv source
archive contains every included TeX file, `main.bbl`, the bibliography, and four
external PDF figures; the workflow figure is drawn in TeX. The standalone ASCII
abstract is below arXiv's 1,920-character limit. This software release is not an
arXiv submission.

## Interpretation

Residual forecasts beat persistence for every asset in both periods, while an
analytic rolling-window update remains competitive. Predictive likelihood gains
from fitted Heston models are mixed. With 20,000 scenarios, mean allocation
variation across seeds is smaller than the mean robustness tilt; return
advantages over matched nominal allocation remain statistically inconclusive.

The high 2025 CAGRs require context: GLD rose 63.7% over the calendar year, and
the historical CVaR benchmark averaged about 90% gold exposure. Its roughly 61%
annualized portfolio CAGR is largely a concentrated gold position. One such
year does not validate broad forecasting or investment superiority.

Predictors use signal-close information. The optimizer marks holdings at the
next adjusted open and assumes immediate rebalancing there. Costs are seven
basis points per risky purchase or sale; cash earns a stylized 3% annually.
Adjusted opens approximate total returns. Slippage, spreads, taxes, and execution
delay are omitted. Bootstrap intervals are descriptive, unadjusted, and
conditional on fitted paths.

## Licence and acknowledgment

The software is [MIT licensed](LICENSE). The manuscript, author-created research
figures, and tables are [CC BY 4.0](paper/LICENSE). Third-party provider data are
excluded from both grants. Sheehan Mathur is the sole author. Anthropic Claude
assisted with Table 1 and parts of the literature review; OpenAI Codex assisted
with implementation, experiments, drafting, checking, and typesetting. See the
manuscript disclosure and `CITATION.cff`.
