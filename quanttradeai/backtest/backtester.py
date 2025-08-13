"""Backtester: vectorised trade simulation utilities.

This module implements simple helpers for running trading strategy
simulations based on pre-generated labels and for calculating basic
performance statistics.

Key Components:
    - :func:`simulate_trades`: run a labelled backtest for one or more symbols
    - :func:`compute_metrics`: derive Sharpe ratio and drawdown statistics

Typical Usage:
    ```python
    from quanttradeai.backtest import simulate_trades, compute_metrics

    results = simulate_trades(df)
    metrics = compute_metrics(results)
    ```
"""

import pandas as pd
from quanttradeai.utils.metrics import sharpe_ratio, max_drawdown
from quanttradeai.trading.risk import apply_stop_loss_take_profit
from quanttradeai.trading.portfolio import PortfolioManager


def _simulate_single(
    df: pd.DataFrame,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    execution: dict | None = None,
) -> pd.DataFrame:
    """Simulate trades for a single symbol with execution effects."""
    data = df.copy()
    if stop_loss_pct is not None or take_profit_pct is not None:
        data = apply_stop_loss_take_profit(data, stop_loss_pct, take_profit_pct)

    exec_cfg = execution or {}
    tc = exec_cfg.get("transaction_costs", {})
    sl = exec_cfg.get("slippage", {})
    liq = exec_cfg.get("liquidity", {})

    ref_col = "Close"
    if sl.get("reference_price", "close") == "mid" and "Mid" in data.columns:
        ref_col = "Mid"
    prices = data[ref_col].astype(float)
    volumes = data.get("Volume", pd.Series(float("inf"), index=data.index))

    position = 0.0
    carry = 0.0
    entry_price: float | None = None
    gross_returns: list[float] = [0.0] * len(data)
    net_returns: list[float] = [0.0] * len(data)
    ledger: list[dict] = []

    for i in range(len(data)):
        price = prices.iloc[i]
        volume = volumes.iloc[i] if i < len(volumes) else float("inf")
        desired = data["label"].iloc[i] + carry

        diff = desired - position
        side = 1 if diff > 0 else -1 if diff < 0 else 0
        qty = abs(diff)
        total_cost = 0.0

        if side != 0 and qty > 0:
            if liq.get("enabled", False):
                max_part = liq.get("max_participation", 0.0)
                max_qty = max_part * float(volume)
                exec_qty = min(qty, max_qty)
                carry = qty - exec_qty
            else:
                exec_qty = qty
                carry = 0.0

            if exec_qty > 0:
                slip_amt = 0.0
                slip_bps = 0.0
                if sl.get("enabled", False) and sl.get("value", 0) > 0:
                    if sl.get("mode", "bps") == "bps":
                        slip_bps = sl["value"]
                        slip_amt = price * slip_bps / 10000
                    else:
                        slip_amt = sl["value"]
                        slip_bps = slip_amt / price * 10000
                fill_price = price + slip_amt if side > 0 else price - slip_amt

                t_cost = 0.0
                if tc.get("enabled", False) and tc.get("value", 0) > 0:
                    if tc.get("mode", "bps") == "bps":
                        t_cost = price * exec_qty * tc["value"] / 10000
                    else:
                        if tc.get("apply_on", "notional") == "shares":
                            t_cost = tc["value"] * exec_qty
                        else:
                            t_cost = tc["value"]

                sl_cost = abs(slip_amt) * exec_qty
                total_cost = t_cost + sl_cost

                gross_pnl = 0.0
                if position != 0 and side != (1 if position > 0 else -1):
                    close_qty = min(abs(position), exec_qty)
                    if position > 0:
                        gross_pnl = (fill_price - entry_price) * close_qty
                    else:
                        gross_pnl = (entry_price - fill_price) * close_qty
                    if exec_qty > close_qty:
                        entry_price = fill_price
                elif position == 0:
                    entry_price = fill_price
                elif side == (1 if position > 0 else -1):
                    entry_price = (
                        entry_price * abs(position) + fill_price * exec_qty
                    ) / (abs(position) + exec_qty)

                position += side * exec_qty
                if position == 0:
                    entry_price = None

                ledger.append(
                    {
                        "timestamp": data.index[i],
                        "side": "buy" if side > 0 else "sell",
                        "qty": exec_qty,
                        "reference_price": price,
                        "fill_price": fill_price,
                        "gross_pnl_contrib": gross_pnl,
                        "transaction_cost": t_cost,
                        "slippage_cost": sl_cost,
                        "costs": total_cost,
                        "slippage_bps_applied": slip_bps,
                        "net_pnl_contrib": gross_pnl - total_cost,
                    }
                )

        if i < len(data) - 1:
            price_next = prices.iloc[i + 1]
            gross_ret = (price_next - price) / price * position
            cost_return = total_cost / price if price else 0.0
            gross_returns[i] = gross_ret
            net_returns[i] = gross_ret - cost_return
    gross_returns[-1] = 0.0
    net_returns[-1] = 0.0
    data["gross_return"] = pd.Series(gross_returns, index=data.index)
    data["strategy_return"] = pd.Series(net_returns, index=data.index)
    data["gross_equity_curve"] = (1 + data["gross_return"]).cumprod()
    data["equity_curve"] = (1 + data["strategy_return"]).cumprod()
    data.attrs["ledger"] = pd.DataFrame(ledger)
    return data


def simulate_trades(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    execution: dict | None = None,
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
        Legacy cost per trade (as fraction of notional). Converted to execution
        config if provided.
    slippage : float, optional
        Legacy slippage per trade (as fraction of notional). Converted to
        execution config if provided.
    execution : dict, optional
        Execution configuration controlling transaction costs, slippage and
        liquidity limits.
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
    exec_cfg = execution.copy() if execution else {}
    if transaction_cost:
        exec_cfg.setdefault("transaction_costs", {})
        exec_cfg["transaction_costs"].update(
            {"enabled": True, "mode": "bps", "value": transaction_cost * 10000}
        )
    if slippage:
        exec_cfg.setdefault("slippage", {})
        exec_cfg["slippage"].update(
            {"enabled": True, "mode": "bps", "value": slippage * 10000}
        )

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
                execution=exec_cfg,
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
            execution=exec_cfg,
        )


def compute_metrics(data: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """Return gross and net performance summary."""
    from quanttradeai.utils.metrics import compute_performance

    return compute_performance(data, risk_free_rate)
