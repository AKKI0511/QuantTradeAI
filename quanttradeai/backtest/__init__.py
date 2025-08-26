"""Backtesting utilities.

This package contains helpers for simulating trades and evaluating
strategy performance.

Public API:
    - :func:`simulate_trades`
    - :func:`compute_metrics`

Quick Start:
    ```python
    from quanttradeai.backtest import simulate_trades
    results = simulate_trades(dataframe)
    ```
"""

from .backtester import simulate_trades, compute_metrics

__all__ = ["simulate_trades", "compute_metrics"]
