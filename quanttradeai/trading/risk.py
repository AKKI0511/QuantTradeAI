"""Risk management utilities.

Contains helpers for applying stop-loss / take-profit rules and
calculating position sizes.

Key Components:
    - :func:`apply_stop_loss_take_profit`
    - :func:`position_size`

Typical Usage:
    ```python
    from quanttradeai.trading import apply_stop_loss_take_profit
    df = apply_stop_loss_take_profit(df, 0.02, 0.04)
    ```
"""

import pandas as pd


def apply_stop_loss_take_profit(
    df: pd.DataFrame,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> pd.DataFrame:
    """Apply stop-loss and take-profit rules to trading signals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``Close`` prices and ``label`` column.
    stop_loss_pct : float, optional
        Stop-loss percentage as a decimal (e.g. ``0.05`` for 5%). If ``None``
        stop-loss is not applied.
    take_profit_pct : float, optional
        Take-profit percentage as a decimal. If ``None`` take-profit is not
        applied.

    Returns
    -------
    pd.DataFrame
        DataFrame with an adjusted ``label`` column after applying rules.
    """
    data = df.copy()
    labels = data["label"].tolist()

    position = 0
    entry_price: float | None = None
    force_flat = False

    for i, price in enumerate(data["Close"]):
        signal = labels[i]

        if position == 0:
            if force_flat:
                if signal == 0:
                    force_flat = False
                labels[i] = 0
                continue
            if signal != 0:
                position = signal
                entry_price = price
        else:
            hit_sl = False
            hit_tp = False
            if position == 1:
                ret = (price - entry_price) / entry_price
                if stop_loss_pct is not None and ret <= -stop_loss_pct:
                    hit_sl = True
                if take_profit_pct is not None and ret >= take_profit_pct:
                    hit_tp = True
            else:  # short position
                ret = (entry_price - price) / entry_price
                if stop_loss_pct is not None and ret <= -stop_loss_pct:
                    hit_sl = True
                if take_profit_pct is not None and ret >= take_profit_pct:
                    hit_tp = True

            if hit_sl or hit_tp:
                position = 0
                entry_price = None
                labels[i] = 0
                signal = 0
                force_flat = True
            else:
                labels[i] = position

            # if a new non-zero signal appears, enter new position
            if position == 0 and signal != 0:
                position = signal
                entry_price = price
            elif position != 0 and signal != position and signal != 0:
                position = signal
                entry_price = price
                labels[i] = signal

    data["label"] = labels
    return data


def position_size(
    capital: float, risk_per_trade: float, stop_loss_pct: float, price: float
) -> int:
    """Calculate position size based on account risk parameters."""
    if price <= 0:
        raise ValueError("price must be positive")
    if stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct must be positive")

    risk_amount = capital * risk_per_trade
    qty = risk_amount / (price * stop_loss_pct)
    return max(int(qty), 0)
