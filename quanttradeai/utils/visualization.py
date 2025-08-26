"""Visualization helpers.

Simple plotting utilities for exploratory analysis of price series and
strategy performance.

Key Components:
    - :func:`plot_price`
    - :func:`plot_performance`
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_price(data: pd.DataFrame, title: str = "Price Chart") -> None:
    """Plot OHLC closing price."""
    data["Close"].plot(figsize=(10, 4))
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.tight_layout()
    plt.show()


def plot_performance(equity_curve: pd.Series, title: str = "Equity Curve") -> None:
    """Plot cumulative returns or equity curve."""
    equity_curve.plot(figsize=(10, 4))
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()
