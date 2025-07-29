import pandas as pd

from quanttradeai.utils.metrics import sharpe_ratio, max_drawdown
from quanttradeai.trading.risk import apply_stop_loss_take_profit
from quanttradeai.trading.portfolio import PortfolioManager


def _simulate_single(
    df: pd.DataFrame,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> pd.DataFrame:
    """Simulate trades for a single symbol."""
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


def simulate_trades(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    portfolio: PortfolioManager | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Simulate trades using label signals.

    Parameters
    ----------
    df : pd.DataFrame or dict[str, pd.DataFrame]
        Single DataFrame or mapping of symbol to DataFrame containing ``Close``
        prices and a ``label`` column where ``1`` indicates a long position,
        ``-1`` a short position and ``0`` no position.
    stop_loss_pct : float or None, optional
        Stop loss percentage applied to each trade. ``None`` disables stop
        losses.
    take_profit_pct : float or None, optional
        Take profit percentage applied to each trade. ``None`` disables take
        profits.
    transaction_cost : float, optional
        Fixed cost applied every time a position is opened or closed.
    slippage : float, optional
        Additional cost applied on each trade to model slippage.
    portfolio : PortfolioManager or None, optional
        Portfolio manager used to allocate capital when backtesting multiple
        symbols. Required if ``df`` is a dictionary.

    Returns
    -------
    pd.DataFrame or dict[str, pd.DataFrame]
        If ``df`` is a single DataFrame, returns that DataFrame with additional
        ``strategy_return`` and ``equity_curve`` columns. If ``df`` is a
        dictionary, returns a dictionary with per-symbol results as well as an
        aggregated ``"portfolio"`` entry containing the combined equity curve.
    """
    if isinstance(df, dict):
        if portfolio is None:
            raise ValueError("portfolio manager required for multiple symbols")
        combined = None
        results: dict[str, pd.DataFrame] = {}
        for symbol, data in df.items():
            res = _simulate_single(
                data,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                transaction_cost=transaction_cost,
                slippage=slippage,
            )
            results[symbol] = res
            qty = portfolio.open_position(symbol, data["Close"].iloc[0], stop_loss_pct)
            weight = (
                (qty * data["Close"].iloc[0]) / portfolio.initial_capital
                if portfolio.initial_capital
                else 0
            )
            ret = res["strategy_return"] * weight
            combined = ret if combined is None else combined.add(ret, fill_value=0.0)
        combined = combined.fillna(0.0)
        portfolio_curve = (1 + combined).cumprod()
        results["portfolio"] = pd.DataFrame(
            {"strategy_return": combined, "equity_curve": portfolio_curve}
        )
        return results
    else:
        return _simulate_single(
            df,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )


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
