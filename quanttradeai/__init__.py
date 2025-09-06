"""Lightweight package initializer for QuantTradeAI.

Only core trading and backtesting utilities are imported to keep the
module usable without optional heavy dependencies."""

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
