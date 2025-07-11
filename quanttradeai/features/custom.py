import pandas as pd


def momentum_score(
    close: pd.Series,
    sma: pd.Series,
    rsi_series: pd.Series,
    macd: pd.Series,
    macd_signal: pd.Series,
) -> pd.Series:
    """Compute a simple momentum score from multiple indicators."""
    score = (
        (close > sma).astype(int) * 0.3
        + (rsi_series > 50).astype(int) * 0.3
        + (macd > macd_signal).astype(int) * 0.4
    )
    return (score - score.mean()) / score.std()


def volatility_breakout(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 20,
    threshold: float = 2.0,
) -> pd.Series:
    """Flag days when price breaks above the previous high plus a threshold."""
    rolling_high = high.shift(1).rolling(lookback).max()
    rolling_low = low.shift(1).rolling(lookback).min()
    breakout = close > (rolling_high + threshold * (rolling_high - rolling_low))
    return breakout.astype(int)
