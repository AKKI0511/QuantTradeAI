from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import os
import pandas as pd


class DataSource(ABC):
    """Abstract interface for price data providers."""

    @abstractmethod
    def fetch(self, symbol: str, start: str, end: str, interval: Optional[str] = None) -> pd.DataFrame:
        """Retrieve OHLCV data for a single symbol."""
        raise NotImplementedError


class YFinanceDataSource(DataSource):
    """DataSource implementation using the yfinance package."""

    def fetch(self, symbol: str, start: str, end: str, interval: Optional[str] = None) -> pd.DataFrame:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        # Default to daily data if no interval specified
        interval = interval or "1d"
        return ticker.history(start=start, end=end, interval=interval)


class AlphaVantageDataSource(DataSource):
    """DataSource implementation for AlphaVantage API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("AlphaVantage API key not provided")
        from alpha_vantage.timeseries import TimeSeries

        self.ts = TimeSeries(key=self.api_key, output_format="pandas")

    def fetch(self, symbol: str, start: str, end: str, interval: Optional[str] = None) -> pd.DataFrame:
        # AlphaVantage supports different intervals: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
        # For now, we'll use daily as default since intraday requires different API calls
        if interval and interval != "1d":
            # For intraday data, we'd need to use different AlphaVantage endpoints
            # This is a simplified implementation
            raise NotImplementedError(f"Interval {interval} not yet supported for AlphaVantage")
        
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
