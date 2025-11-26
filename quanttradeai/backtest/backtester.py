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
from quanttradeai.trading.risk import apply_stop_loss_take_profit
from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.trading.drawdown_guard import DrawdownGuard
from .intrabar import BrownianParams, generate_gbm_ticks


def _simulate_single(
    df: pd.DataFrame,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    execution: dict | None = None,
    drawdown_guard: DrawdownGuard | None = None,
) -> pd.DataFrame:
    """Simulate trades for a single symbol with execution effects."""
    data = df.copy()
    if stop_loss_pct is not None or take_profit_pct is not None:
        data = apply_stop_loss_take_profit(data, stop_loss_pct, take_profit_pct)

    exec_cfg = execution or {}
    tc = exec_cfg.get("transaction_costs", {})
    sl = exec_cfg.get("slippage", {})
    liq = exec_cfg.get("liquidity", {})
    impact_cfg = exec_cfg.get("impact", {})
    borrow_cfg = exec_cfg.get("borrow_fee", {})
    intrabar_cfg = exec_cfg.get("intrabar", {})

    impact_calc = None
    if impact_cfg.get("enabled", False):
        from .impact import MODEL_MAP, ImpactCalculator, DynamicSpreadModel

        model_cls = MODEL_MAP.get(
            impact_cfg.get("model", "linear"), MODEL_MAP["linear"]
        )
        model_params = {k: v for k, v in impact_cfg.items() if k in {"alpha", "beta"}}
        # ``gamma`` is only relevant for the Almgren-Chriss model; the
        # configuration schema sets it to ``None`` by default, which would raise
        # ``TypeError`` if passed to other models. Only include it when provided
        # and when the chosen model expects it.
        gamma = impact_cfg.get("gamma")
        if gamma is not None and model_cls is MODEL_MAP["almgren_chriss"]:
            model_params["gamma"] = gamma
        model = model_cls(**model_params)
        spread_model = None
        if impact_cfg.get("spread_model"):
            sm = impact_cfg["spread_model"]
            spread_model = DynamicSpreadModel(
                base=sm.get("base", 0.0),
                vol_coeff=sm.get("vol_coeff", 0.0),
                volume_coeff=sm.get("volume_coeff", 0.0),
                tod=sm.get("tod"),
            )
        impact_calc = ImpactCalculator(
            model=model,
            decay=impact_cfg.get("decay", 0.0),
            decay_volume_coeff=impact_cfg.get("decay_volume_coeff", 0.0),
            spread=impact_cfg.get("spread", 0.0),
            spread_model=spread_model,
            alpha_buy=impact_cfg.get("alpha_buy"),
            alpha_sell=impact_cfg.get("alpha_sell"),
            beta_buy=impact_cfg.get("beta_buy"),
            beta_sell=impact_cfg.get("beta_sell"),
            cross_alpha=impact_cfg.get("cross_alpha", 0.0),
            cross_beta=impact_cfg.get("cross_beta", 0.0),
            horizon_decay=impact_cfg.get("horizon_decay", 0.0),
        )

    ref_col = "Close"
    if sl.get("reference_price", "close") == "mid" and "Mid" in data.columns:
        ref_col = "Mid"
    prices = data[ref_col].astype(float)
    volumes = data.get("Volume", pd.Series(float("inf"), index=data.index))
    order_types = data.get(
        "order_type", pd.Series("market", index=data.index, dtype="object")
    )
    limit_prices = data.get("limit_price", pd.Series(float("nan"), index=data.index))
    stop_prices = data.get("stop_price", pd.Series(float("nan"), index=data.index))
    vol_series = data.get("Volatility", pd.Series(0.0, index=data.index, dtype=float))

    position = 0.0
    carry = 0.0
    entry_price: float | None = None
    gross_returns: list[float] = [0.0] * len(data)
    net_returns: list[float] = [0.0] * len(data)
    ledger: list[dict] = []
    equity_value = 1.0
    last_timestamp = data.index[0] if not data.empty else None
    updated_portfolio = False

    for i in range(len(data)):
        price = prices.iloc[i]
        volume = volumes.iloc[i] if i < len(volumes) else float("inf")
        label = data["label"].iloc[i]

        size_multiplier = 1.0
        halt_trading = False
        emergency_liquidate = False
        if drawdown_guard is not None:
            drawdown_guard.check_drawdown_limits()
            size_multiplier = drawdown_guard.get_position_size_multiplier()
            halt_trading = drawdown_guard.should_halt_trading()
            emergency_liquidate = drawdown_guard.should_emergency_liquidate()

        if carry and label * carry <= 0:
            carry = 0.0
        if emergency_liquidate:
            desired = 0.0
            carry = 0.0
        else:
            scaled_label = label * size_multiplier
            if halt_trading:
                desired = 0.0
                carry = 0.0
            else:
                desired = scaled_label + carry

        diff = desired - position
        side = 1 if diff > 0 else -1 if diff < 0 else 0
        qty = abs(diff)
        total_cost = 0.0

        if side != 0 and qty > 0:
            order_type = order_types.iloc[i]
            limit_p = limit_prices.iloc[i]
            stop_p = stop_prices.iloc[i]
            exec_qty = qty
            if order_type == "limit":
                if not (
                    (side > 0 and price <= limit_p) or (side < 0 and price >= limit_p)
                ):
                    carry += side * qty
                    exec_qty = 0.0
            elif order_type == "stop":
                triggered = (side > 0 and price >= stop_p) or (
                    side < 0 and price <= stop_p
                )
                if not triggered:
                    carry += side * qty
                    exec_qty = 0.0

            if exec_qty > 0:
                if liq.get("enabled", False):
                    max_part = liq.get("max_participation", 0.0)
                    max_qty = max_part * float(volume)
                    exec_qty = min(exec_qty, max_qty)
                    carry = side * (qty - exec_qty)
                else:
                    carry = 0.0

                depth = liq.get("order_book_depth")
                if depth:
                    fill_prob = min(1.0, float(volume) / depth)
                    filled = exec_qty * fill_prob
                    carry += side * (exec_qty - filled)
                    exec_qty = filled

                slip_amt = 0.0
                slip_bps = 0.0
                if sl.get("enabled", False) and sl.get("value", 0) > 0:
                    if sl.get("mode", "bps") == "bps":
                        slip_bps = sl["value"]
                        slip_amt = price * slip_bps / 10000
                    else:
                        slip_amt = sl["value"]
                        slip_bps = slip_amt / price * 10000

                impact_cost = 0.0
                impact_temp = 0.0
                impact_perm = 0.0
                impact_adjust = 0.0
                if impact_calc is not None:
                    adv = impact_cfg.get("average_daily_volume", float(volume))
                    adv *= impact_cfg.get("liquidity_scale", 1.0)
                    imp = impact_calc.impact_cost(
                        exec_qty,
                        adv,
                        side=side,
                        volatility=vol_series.iloc[i],
                        volume=float(volume),
                        timestamp=data.index[i],
                    )
                    impact_cost = imp["total"]
                    impact_temp = imp["temp"] * exec_qty
                    impact_perm = imp["perm"] * exec_qty
                    impact_adjust = (imp["temp"] + imp["spread"]) * (
                        1 if side > 0 else -1
                    )
                # Intrabar tick-level fills
                liquidity_part = 0.0
                tick_fills = 0
                ticks = None
                if intrabar_cfg.get("enabled", False):
                    tick_col = intrabar_cfg.get("tick_column", "ticks")
                    if tick_col in data.columns:
                        ticks = data[tick_col].iloc[i]
                    if not ticks:
                        params = BrownianParams(
                            drift=intrabar_cfg.get("drift", 0.0),
                            volatility=intrabar_cfg.get("volatility", 0.0),
                            ticks=intrabar_cfg.get("synthetic_ticks", 1),
                        )
                        ticks = generate_gbm_ticks(
                            price,
                            float(volume),
                            params,
                            seed=i,
                        )
                if ticks is not None:
                    tick_vol = sum(t.get("volume", 0.0) for t in ticks)
                    if tick_vol < exec_qty:
                        carry += side * (exec_qty - tick_vol)
                        exec_qty = tick_vol
                    filled = 0.0
                    vwap = 0.0
                    for t in ticks:
                        if filled >= exec_qty:
                            break
                        take = min(exec_qty - filled, t.get("volume", 0.0))
                        if take > 0:
                            vwap += t.get("price", price) * take
                            filled += take
                            tick_fills += 1
                    base_price = vwap / exec_qty if exec_qty > 0 else price
                    fill_price = (
                        base_price + slip_amt * (1 if side > 0 else -1) + impact_adjust
                    )
                    liquidity_part = exec_qty / tick_vol if tick_vol > 0 else 0.0
                else:
                    base_price = price
                    fill_price = (
                        base_price + slip_amt * (1 if side > 0 else -1) + impact_adjust
                    )
                    liquidity_part = exec_qty / float(volume) if float(volume) else 0.0
                    tick_fills = 0

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
                total_cost = t_cost + sl_cost + impact_cost

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
                        "impact_temp_cost": impact_temp,
                        "impact_perm_cost": impact_perm,
                        "impact_cost": impact_cost,
                        "costs": total_cost,
                        "slippage_bps_applied": slip_bps,
                        "net_pnl_contrib": gross_pnl - total_cost,
                        "borrow_fee": 0.0,
                        "liquidity_participation": liquidity_part,
                        "tick_fills": tick_fills,
                        "intrabar": intrabar_cfg.get("enabled", False),
                        "order_type": order_type,
                    }
                )
                if drawdown_guard is not None and exec_qty > 0:
                    notional = exec_qty * fill_price
                    drawdown_guard.record_trade(notional, data.index[i])

        borrow_cost = 0.0
        if (
            borrow_cfg.get("enabled", False)
            and borrow_cfg.get("rate_bps", 0) > 0
            and position < 0
        ):
            borrow_cost = abs(position) * price * borrow_cfg["rate_bps"] / 10000
            total_cost += borrow_cost
            ledger.append(
                {
                    "timestamp": data.index[i],
                    "side": "borrow_fee",
                    "qty": abs(position),
                    "reference_price": price,
                    "fill_price": price,
                    "gross_pnl_contrib": 0.0,
                    "transaction_cost": 0.0,
                    "slippage_cost": 0.0,
                    "impact_temp_cost": 0.0,
                    "impact_perm_cost": 0.0,
                    "impact_cost": 0.0,
                    "costs": borrow_cost,
                    "slippage_bps_applied": 0.0,
                    "net_pnl_contrib": -borrow_cost,
                    "borrow_fee": borrow_cost,
                    "liquidity_participation": 0.0,
                    "tick_fills": 0,
                    "intrabar": False,
                }
            )

        if i < len(data) - 1:
            price_next = prices.iloc[i + 1]
            gross_ret = (price_next - price) / price * position
            cost_return = total_cost / price if price else 0.0
            gross_returns[i] = gross_ret
            net_returns[i] = gross_ret - cost_return
            if drawdown_guard is not None:
                equity_value *= 1 + net_returns[i]
                last_timestamp = data.index[i + 1]
                drawdown_guard.update_portfolio_value(
                    equity_value,
                    last_timestamp,
                )
                updated_portfolio = True
    gross_returns[-1] = 0.0
    net_returns[-1] = 0.0
    data["gross_return"] = pd.Series(gross_returns, index=data.index)
    data["strategy_return"] = pd.Series(net_returns, index=data.index)
    data["gross_equity_curve"] = (1 + data["gross_return"]).cumprod()
    data["equity_curve"] = (1 + data["strategy_return"]).cumprod()
    data.attrs["ledger"] = pd.DataFrame(ledger)
    if (
        drawdown_guard is not None
        and not updated_portfolio
        and last_timestamp is not None
    ):
        drawdown_guard.update_portfolio_value(equity_value, last_timestamp)
    return data


def simulate_trades(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    execution: dict | None = None,
    portfolio: PortfolioManager | None = None,
    drawdown_guard: DrawdownGuard | None = None,
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
    exec_cfg_input = execution or {}
    exec_cfg_global: dict = {}
    exec_cfg_by_symbol: dict[str, dict] | None = None

    def _apply_legacy_overrides(cfg: dict) -> dict:
        updated = cfg.copy()
        if transaction_cost:
            updated.setdefault("transaction_costs", {})
            updated["transaction_costs"].update(
                {
                    "enabled": True,
                    "mode": "bps",
                    "value": transaction_cost * 10000,
                }
            )
        if slippage:
            updated.setdefault("slippage", {})
            updated["slippage"].update(
                {"enabled": True, "mode": "bps", "value": slippage * 10000}
            )
        return updated

    if isinstance(df, dict) and isinstance(exec_cfg_input, dict):
        symbol_keys = {k for k in exec_cfg_input if k in df}
        if symbol_keys and symbol_keys == set(exec_cfg_input.keys()):
            exec_cfg_by_symbol = {
                symbol: _apply_legacy_overrides(cfg)
                for symbol, cfg in exec_cfg_input.items()
            }
        else:
            exec_cfg_global = _apply_legacy_overrides(exec_cfg_input.copy())
    elif isinstance(exec_cfg_input, dict):
        exec_cfg_global = _apply_legacy_overrides(exec_cfg_input.copy())

    if isinstance(df, dict):
        if portfolio is None:
            raise ValueError("portfolio manager required for multiple symbols")
        combined = None
        results: dict[str, pd.DataFrame] = {}
        for symbol, data in df.items():
            symbol_exec_cfg = (
                exec_cfg_by_symbol.get(symbol, exec_cfg_global)
                if exec_cfg_by_symbol is not None
                else exec_cfg_global
            )
            res = _simulate_single(
                data,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                execution=symbol_exec_cfg,
                drawdown_guard=drawdown_guard,
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
        if drawdown_guard is not None:
            curve = results["portfolio"]["equity_curve"]
            for t, v in zip(curve.index, curve):
                drawdown_guard.update_portfolio_value(float(v), t)
        return results
    else:
        res = _simulate_single(
            df,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            execution=exec_cfg_global,
            drawdown_guard=drawdown_guard,
        )
        if drawdown_guard is not None:
            curve = res["equity_curve"]
            for t, v in zip(curve.index, curve):
                drawdown_guard.update_portfolio_value(float(v), t)
        return res


def compute_metrics(data: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """Return gross and net performance summary."""
    from quanttradeai.utils.metrics import compute_performance

    return compute_performance(data, risk_free_rate)
