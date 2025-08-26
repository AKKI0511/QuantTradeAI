"""Technical indicator wrappers.

Thin wrappers around ``pandas-ta`` functions used by the project.  These
helpers provide a consistent namespace for commonly used indicators.

Key Components:
    - :func:`sma`
    - :func:`ema`
    - :func:`rsi`
    - :func:`macd`
    - :func:`stochastic`

Typical Usage:
    ```python
    from quanttradeai.features import technical as ta
    sma_fast = ta.sma(close, 20)
    ```
"""

import pandas as pd
import pandas_ta as ta


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return ta.sma(series, length=period)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return ta.ema(series, length=period)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    return ta.rsi(series, length=period)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD indicator returning macd, signal and histogram columns"""
    df = ta.macd(series, fast=fast, slow=slow, signal=signal)
    return pd.DataFrame(
        {
            "macd": df[f"MACD_{fast}_{slow}_{signal}"],
            "signal": df[f"MACDs_{fast}_{slow}_{signal}"],
            "hist": df[f"MACDh_{fast}_{slow}_{signal}"],
        }
    )


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator with parameter-agnostic column selection.

    ``pandas_ta.stoch`` encodes the parameters in its column names
    (e.g. ``STOCHk_14_3_3``). This helper renames the first two returned
    columns to a stable ``stoch_k`` and ``stoch_d`` regardless of the
    ``k``/``d`` values used.

    Example
    -------
    >>> import pandas as pd
    >>> h = pd.Series([2, 3, 4, 5, 6])
    >>> l = pd.Series([1, 1, 2, 2, 3])
    >>> c = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> out = stochastic(h, l, c, k=10, d=3)
    >>> set(["stoch_k", "stoch_d"]).issubset(out.columns)
    True
    """
    df = ta.stoch(high, low, close, k=k, d=d)
    # Be resilient to pandas-ta naming: pick the first k/d columns returned
    k_col = next(
        (c for c in df.columns if c.lower().startswith("stochk")), df.columns[0]
    )
    d_col = next(
        (c for c in df.columns if c.lower().startswith("stochd")),
        df.columns[min(1, len(df.columns) - 1)],
    )
    return pd.DataFrame({"stoch_k": df[k_col], "stoch_d": df[d_col]})
