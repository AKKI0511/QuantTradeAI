import pandas as pd

from utils.metrics import sharpe_ratio, max_drawdown
from trading.risk import apply_stop_loss_take_profit


def simulate_trades(
    df: pd.DataFrame,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> pd.DataFrame:
    """Simulate trades using label signals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``Close`` prices and a ``label`` column where
        1 indicates a long position, -1 a short position and 0 no position.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional ``strategy_return`` and
        ``equity_curve`` columns.
    """
    data = df.copy()
    if stop_loss_pct is not None or take_profit_pct is not None:
        data = apply_stop_loss_take_profit(data, stop_loss_pct, take_profit_pct)
    data["price_return"] = data["Close"].pct_change()
    data["strategy_return"] = data["price_return"].shift(-1) * data["label"]
    data["strategy_return"] = data["strategy_return"].fillna(0.0)
    data["equity_curve"] = (1 + data["strategy_return"]).cumprod()
    return data


def compute_metrics(data: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """Calculate basic performance metrics for a strategy.

    Parameters
    ----------
    data : pd.DataFrame
        Output from :func:`simulate_trades` containing ``strategy_return`` and
        ``equity_curve`` columns.
    risk_free_rate : float, optional
        Annual risk free rate used in Sharpe ratio, by default 0.0.

    Returns
    -------
    dict
        Dictionary with ``cumulative_return``, ``sharpe_ratio`` and
        ``max_drawdown`` keys.
    """
    returns = data["strategy_return"]
    equity = data["equity_curve"]
    cumulative_return = equity.iloc[-1] - 1
    sharpe = sharpe_ratio(returns, risk_free_rate)
    mdd = max_drawdown(equity)
    return {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
    }
