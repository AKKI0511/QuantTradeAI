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
    """Calculate the annualized Sharpe ratio.

    The function guards against divisions by zero and empty inputs by
    returning ``0.0`` when the input has no variance or no observations.

    Example
    -------
    >>> import pandas as pd
    >>> r = pd.Series([0.01, -0.005, 0.0, 0.007])
    >>> round(sharpe_ratio(r), 3) > 0
    True
    """
    if returns is None or len(returns) == 0:
        return 0.0
    excess = returns - risk_free_rate / 252
    std = float(excess.std())
    if std == 0 or np.isnan(std):
        return 0.0
    mean = float(excess.mean())
    return float(np.sqrt(252) * mean / std)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    return drawdown.min()


def cagr(equity_curve: pd.Series) -> float:
    """Compound annual growth rate assuming daily data.

    Returns a Python float to ensure JSON-safe serialization.
    """
    if equity_curve.empty:
        return 0.0
    periods = len(equity_curve)
    years = periods / 252
    final = float(equity_curve.iloc[-1])
    return float(final ** (1 / years) - 1)


def compute_performance(data: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """Return gross and net performance statistics.

    Ensures all values are JSON-serializable scalars. Large time series are
    intentionally omitted from the summary to keep artifacts compact.
    """
    returns = data.get("gross_return", data["strategy_return"])  # pd.Series
    net_returns = data["strategy_return"]  # pd.Series
    equity = data.get("gross_equity_curve", data["equity_curve"])  # pd.Series
    net_equity = data["equity_curve"]  # pd.Series

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
        "gross_pnl": float(float(equity.iloc[-1]) - 1.0),
        "total_costs": float(total_costs),
        "total_slippage_cost": float(total_slippage),
        "net_pnl": float(float(net_equity.iloc[-1]) - 1.0),
        "gross_sharpe": float(sharpe_ratio(returns, risk_free_rate)),
        "net_sharpe": float(sharpe_ratio(net_returns, risk_free_rate)),
        "gross_cagr": float(cagr(equity)),
        "net_cagr": float(cagr(net_equity)),
        "gross_mdd": float(max_drawdown(equity)),
        "net_mdd": float(max_drawdown(net_equity)),
    }
    # backward-compatible scalar aliases
    summary["cumulative_return"] = summary["gross_pnl"]
    summary["sharpe_ratio"] = summary["gross_sharpe"]
    summary["max_drawdown"] = summary["gross_mdd"]
    return summary
