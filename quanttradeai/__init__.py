"""QuantTradeAI
=================

High-level interface for the QuantTradeAI toolkit.  The package bundles
data acquisition, feature engineering, model training and backtesting
utilities for quantitative trading research.

Public API:
    - ``DataSource`` and concrete implementations
    - ``DataLoader`` and ``DataProcessor``
    - ``MomentumClassifier`` model
    - ``PortfolioManager`` and risk helpers
    - ``simulate_trades`` and ``compute_metrics`` for backtesting

Quick Start:
    ```python
    from quanttradeai import DataLoader, DataProcessor, MomentumClassifier

    loader = DataLoader()
    data = loader.fetch_data()
    processor = DataProcessor()
    processed = {s: processor.process_data(df) for s, df in data.items()}
    model = MomentumClassifier()
    ```
"""

from .data.datasource import (
    DataSource,
    YFinanceDataSource,
    AlphaVantageDataSource,
    WebSocketDataSource,
)
from .data.loader import DataLoader
from .data.processor import DataProcessor
from .models.classifier import MomentumClassifier
from .trading.portfolio import PortfolioManager
from .trading.risk import apply_stop_loss_take_profit, position_size
from .backtest.backtester import simulate_trades, compute_metrics

__all__ = [
    "DataSource",
    "YFinanceDataSource",
    "AlphaVantageDataSource",
    "WebSocketDataSource",
    "DataLoader",
    "DataProcessor",
    "MomentumClassifier",
    "PortfolioManager",
    "apply_stop_loss_take_profit",
    "position_size",
    "simulate_trades",
    "compute_metrics",
]
