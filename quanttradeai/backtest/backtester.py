import pandas as pd

from quanttradeai.utils.metrics import sharpe_ratio, max_drawdown
from quanttradeai.trading.risk import apply_stop_loss_take_profit
from quanttradeai.trading.portfolio import Portfolio


def _simulate_portfolio(
    data: dict[str, pd.DataFrame],
    portfolio: Portfolio,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> pd.DataFrame:
    trade_cost = transaction_cost + slippage
    processed = {}
    for sym, df in data.items():
        d = df.copy()
        if stop_loss_pct is not None or take_profit_pct is not None:
            d = apply_stop_loss_take_profit(d, stop_loss_pct, take_profit_pct)
        processed[sym] = d

    length = min(len(df) for df in processed.values())
    index = list(processed.values())[0].index[:length]
    equity = []
    returns = []
    for i in range(length):
        prices = {sym: df["Close"].iloc[i] for sym, df in processed.items()}
        for sym, df in processed.items():
            signal = df["label"].iloc[i]
            prev = df["label"].iloc[i - 1] if i > 0 else 0
            if signal != prev:
                if prev != 0:
                    portfolio.close(sym, prices[sym], trade_cost)
                if signal != 0:
                    direction = 1 if signal > 0 else -1
                    portfolio.allocate(
                        sym, prices[sym], stop_loss_pct or 0.01, direction, trade_cost
                    )
        portfolio.update_value(prices)
        value = portfolio.total_value
        if i == 0:
            base = value
            returns.append(0.0)
        else:
            ret = (value - equity[-1]) / equity[-1]
            returns.append(ret)
        equity.append(value)

    equity_curve = [v / base for v in equity]
    return pd.DataFrame(
        {"strategy_return": returns, "equity_curve": equity_curve}, index=index
    )


def simulate_trades(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    portfolio: Portfolio | None = None,
) -> pd.DataFrame:
    """Simulate trades using label signals.

    Parameters
    ----------
    df : pd.DataFrame or dict
        DataFrame containing ``Close`` prices and a ``label`` column where
        1 indicates a long position, -1 a short position and 0 no position.
        If a dictionary is provided, each value should be such a DataFrame and
        ``portfolio`` must be supplied.

    transaction_cost : float, optional
        Fixed cost applied every time a position is opened or closed.
    slippage : float, optional
        Additional cost applied on each trade to model slippage.
    portfolio : Portfolio, optional
        Portfolio instance used when backtesting multiple symbols.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional ``strategy_return`` and
        ``equity_curve`` columns.
    """
    if isinstance(df, dict):
        if portfolio is None:
            raise ValueError("portfolio instance required for multi-symbol data")
        return _simulate_portfolio(
            df,
            portfolio,
            stop_loss_pct,
            take_profit_pct,
            transaction_cost,
            slippage,
        )

    data = df.copy()
    if stop_loss_pct is not None or take_profit_pct is not None:
        data = apply_stop_loss_take_profit(data, stop_loss_pct, take_profit_pct)
    data["price_return"] = data["Close"].pct_change()
    data["strategy_return"] = data["price_return"].shift(-1) * data["label"]
    data["strategy_return"] = data["strategy_return"].fillna(0.0)

    trade_cost = transaction_cost + slippage
    if trade_cost > 0:
        trades = data["label"].diff().abs()
        trades.iloc[0] = abs(data["label"].iloc[0])
        data["strategy_return"] -= trades * trade_cost

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
