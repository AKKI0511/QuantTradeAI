from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import os
import pandas as pd


class DataSource(ABC):
    """Abstract interface for price data providers."""

    @abstractmethod
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Retrieve OHLCV data for a single symbol."""
        raise NotImplementedError


class YFinanceDataSource(DataSource):
    """DataSource implementation using the yfinance package."""

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        return ticker.history(start=start, end=end)


class AlphaVantageDataSource(DataSource):
    """DataSource implementation for AlphaVantage API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("AlphaVantage API key not provided")
        from alpha_vantage.timeseries import TimeSeries

        self.ts = TimeSeries(key=self.api_key, output_format="pandas")

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        data, _ = self.ts.get_daily_adjusted(symbol=symbol, outputsize="full")
        data.index = pd.to_datetime(data.index)
        data = data.loc[start:end]
        data = data.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "6. volume": "Volume",
            }
        )
        return data[["Open", "High", "Low", "Close", "Volume"]]
