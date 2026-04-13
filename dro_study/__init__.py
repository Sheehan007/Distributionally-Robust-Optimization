"""Modular toolkit for the DRO study project."""

from .backtest import TradeRecord, run_backtest
from .config import StudyConfig, default_study_config
from .data import MarketObservation, generate_synthetic_market_data, load_market_csv
from .reporting import generate_report, write_outputs

__all__ = [
    "MarketObservation",
    "StudyConfig",
    "TradeRecord",
    "default_study_config",
    "generate_report",
    "generate_synthetic_market_data",
    "load_market_csv",
    "run_backtest",
    "write_outputs",
]
