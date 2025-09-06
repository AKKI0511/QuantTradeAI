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

# Lazily import optional dependencies to keep lightweight usage possible
from .data.loader import DataLoader
from .data.processor import DataProcessor

try:  # pragma: no cover - optional heavy dependency
    from .models.classifier import MomentumClassifier
except Exception:  # pragma: no cover - tolerate missing ML libs
    MomentumClassifier = None  # type: ignore[assignment]
from .trading.portfolio import PortfolioManager
from .trading.risk import apply_stop_loss_take_profit, position_size
from .backtest import (
    simulate_trades,
    compute_metrics,
    MarketImpactModel,
    LinearImpactModel,
    SquareRootImpactModel,
    AlmgrenChrissModel,
    ImpactCalculator,
    DynamicSpreadModel,
    BacktestEngine,
)

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
    "MarketImpactModel",
    "LinearImpactModel",
    "SquareRootImpactModel",
    "AlmgrenChrissModel",
    "ImpactCalculator",
    "DynamicSpreadModel",
    "BacktestEngine",
]
