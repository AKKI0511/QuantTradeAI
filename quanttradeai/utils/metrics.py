"""Metric helper functions.

Convenience wrappers around common evaluation metrics used by the
framework.

Key Components:
    - :func:`classification_metrics`
    - :func:`sharpe_ratio`
    - :func:`max_drawdown`
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return basic classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio."""
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    return drawdown.min()


def cagr(equity_curve: pd.Series) -> float:
    """Compound annual growth rate assuming daily data."""
    if equity_curve.empty:
        return 0.0
    periods = len(equity_curve)
    years = periods / 252
    final = equity_curve.iloc[-1]
    return final ** (1 / years) - 1


def compute_performance(data: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """Return gross and net performance statistics."""
    returns = data.get("gross_return", data["strategy_return"])
    net_returns = data["strategy_return"]
    equity = data.get("gross_equity_curve", data["equity_curve"])
    net_equity = data["equity_curve"]

    ledger = data.attrs.get("ledger")
    if ledger is not None and not ledger.empty:
        total_costs = float(
            (ledger["transaction_cost"] / ledger["reference_price"]).sum()
        )
        total_slippage = float(
            (ledger["slippage_cost"] / ledger["reference_price"]).sum()
        )
    else:
        total_costs = 0.0
        total_slippage = 0.0

    summary = {
        "gross_pnl": float(equity.iloc[-1] - 1),
        "total_costs": total_costs,
        "total_slippage_cost": total_slippage,
        "net_pnl": float(net_equity.iloc[-1] - 1),
        "gross_sharpe": sharpe_ratio(returns, risk_free_rate),
        "net_sharpe": sharpe_ratio(net_returns, risk_free_rate),
        "gross_cagr": cagr(equity),
        "net_cagr": cagr(net_equity),
        "gross_mdd": max_drawdown(equity),
        "net_mdd": max_drawdown(net_equity),
    }
    # backward-compatible keys
    summary["cumulative_return"] = summary["gross_pnl"]
    summary["sharpe_ratio"] = summary["gross_sharpe"]
    summary["max_drawdown"] = summary["gross_mdd"]
    summary["net_returns_series"] = net_returns
    summary["gross_returns_series"] = returns
    return summary
