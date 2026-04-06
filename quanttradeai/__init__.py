"""Lazy public API for QuantTradeAI."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MAP = {
    "DataSource": ("quanttradeai.data.datasource", "DataSource"),
    "YFinanceDataSource": ("quanttradeai.data.datasource", "YFinanceDataSource"),
    "AlphaVantageDataSource": (
        "quanttradeai.data.datasource",
        "AlphaVantageDataSource",
    ),
    "WebSocketDataSource": ("quanttradeai.data.datasource", "WebSocketDataSource"),
    "DataLoader": ("quanttradeai.data.loader", "DataLoader"),
    "DataProcessor": ("quanttradeai.data.processor", "DataProcessor"),
    "AgentDecision": ("quanttradeai.agents", "AgentDecision"),
    "AgentSimulationState": ("quanttradeai.agents", "AgentSimulationState"),
    "BaseStrategy": ("quanttradeai.agents", "BaseStrategy"),
    "RuleAgentStrategy": ("quanttradeai.agents", "RuleAgentStrategy"),
    "MomentumClassifier": ("quanttradeai.models.classifier", "MomentumClassifier"),
    "PortfolioManager": ("quanttradeai.trading.portfolio", "PortfolioManager"),
    "apply_stop_loss_take_profit": (
        "quanttradeai.trading.risk",
        "apply_stop_loss_take_profit",
    ),
    "position_size": ("quanttradeai.trading.risk", "position_size"),
    "simulate_trades": ("quanttradeai.backtest", "simulate_trades"),
    "compute_metrics": ("quanttradeai.backtest", "compute_metrics"),
    "MarketImpactModel": ("quanttradeai.backtest", "MarketImpactModel"),
    "LinearImpactModel": ("quanttradeai.backtest", "LinearImpactModel"),
    "SquareRootImpactModel": ("quanttradeai.backtest", "SquareRootImpactModel"),
    "AlmgrenChrissModel": ("quanttradeai.backtest", "AlmgrenChrissModel"),
    "ImpactCalculator": ("quanttradeai.backtest", "ImpactCalculator"),
    "DynamicSpreadModel": ("quanttradeai.backtest", "DynamicSpreadModel"),
    "BacktestEngine": ("quanttradeai.backtest", "BacktestEngine"),
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'quanttradeai' has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    try:
        module = import_module(module_name)
        value = getattr(module, attr_name)
    except Exception:
        if name == "MomentumClassifier":  # pragma: no cover - optional dependency path
            value = None
        else:
            raise
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
