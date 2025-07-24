from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import os
import pandas as pd


class DataSource(ABC):
    """Abstract interface for price data providers."""

    @abstractmethod
    def fetch(
        self, symbol: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Retrieve OHLCV data for a single symbol.

        Parameters
        ----------
        symbol : str
            Ticker symbol to fetch data for.
        start : str
            Start date for the data range.
        end : str
            End date for the data range.
        interval : str, optional
            Data interval (e.g. ``"1d"`` or ``"1h"``). Implementations
            should document the supported values.
        """
        raise NotImplementedError


class YFinanceDataSource(DataSource):
    """DataSource implementation using the yfinance package.

    Supported intervals include ``"1m"``, ``"2m"``, ``"5m"``, ``"15m"``, ``"30m"``,
    ``"60m"``/``"1h"``, ``"1d"``, ``"5d"``, ``"1wk"``, ``"1mo"`` and ``"3mo"``.
    """

    def fetch(
        self, symbol: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        return ticker.history(start=start, end=end, interval=interval)


class AlphaVantageDataSource(DataSource):
    """DataSource implementation for AlphaVantage API.

    Supported intervals are ``"1min"``, ``"5min"``, ``"15min"``, ``"30min"``,
    ``"60min"``/``"1h"`` for intraday data and ``"1d"`` for daily data.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("AlphaVantage API key not provided")
        from alpha_vantage.timeseries import TimeSeries

        self.ts = TimeSeries(key=self.api_key, output_format="pandas")

    def fetch(
        self, symbol: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        if interval in {"1d", "daily"}:
            data, _ = self.ts.get_daily_adjusted(symbol=symbol, outputsize="full")
        else:
            # map common aliases like "1h" -> "60min"
            mapping = {"1h": "60min"}
            av_interval = mapping.get(interval, interval)
            supported = {"1min", "5min", "15min", "30min", "60min"}
            if av_interval not in supported:
                raise ValueError(
                    f"Interval '{interval}' not supported by AlphaVantage"
                )
            data, _ = self.ts.get_intraday(
                symbol=symbol, interval=av_interval, outputsize="full"
            )

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
