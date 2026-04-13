"""CLI entry point for the DRO study package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dro_study import (
    default_study_config,
    generate_report,
    generate_synthetic_market_data,
    load_market_csv,
    run_backtest,
    write_outputs,
)
from dro_study.regimes import attach_regimes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the regime-aware Heston plus adaptive DRO study pipeline."
    )
    parser.add_argument("--input-csv", type=str, default="", help="Optional CSV with date/close/vix/cpi/fed_funds.")
    parser.add_argument("--output-dir", type=str, default="dro_outputs", help="Folder for JSON and CSV outputs.")
    parser.add_argument("--demo-days", type=int, default=650, help="Synthetic sample length when no CSV is supplied.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the synthetic dataset.")
    parser.add_argument("--train-window", type=int, default=126, help="Rolling estimation window.")
    parser.add_argument("--rebalance-every", type=int, default=5, help="Holding period and rebalance frequency.")
    parser.add_argument("--scenario-count", type=int, default=250, help="Monte Carlo scenario count per rebalance.")
    parser.add_argument("--weight-step", type=float, default=0.05, help="Allocation grid spacing.")
    parser.add_argument(
        "--disable-sensitivity",
        action="store_true",
        help="Skip sensitivity reruns if you only want the base backtest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_study_config()
    config.backtest.train_window = args.train_window
    config.backtest.rebalance_every = args.rebalance_every
    config.simulation.horizon_days = args.rebalance_every
    config.simulation.scenario_count = args.scenario_count
    config.optimizer.weight_step = args.weight_step

    if args.input_csv:
        observations = load_market_csv(args.input_csv)
    else:
        observations = generate_synthetic_market_data(days=args.demo_days, seed=args.seed)

    attach_regimes(observations)
    strategy_records = run_backtest(observations, config)
    report = generate_report(
        observations=observations,
        strategy_records=strategy_records,
        config=config,
        include_sensitivity=not args.disable_sensitivity,
    )
    write_outputs(Path(args.output_dir), report, strategy_records)

    print(json.dumps(report["headline"], indent=2))
    print(f"Outputs written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
