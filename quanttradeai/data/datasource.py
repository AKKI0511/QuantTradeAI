"""Market data sources.

This module defines the :class:`DataSource` interface along with concrete
implementations for Yahoo! Finance, Alpha Vantage and generic WebSocket
feeds.

Key Components:
    - :class:`DataSource`: abstract base for all data providers
    - :class:`YFinanceDataSource`: wrapper around ``yfinance``
    - :class:`AlphaVantageDataSource`: uses the Alpha Vantage REST API
    - :class:`WebSocketDataSource`: asynchronous streaming interface

Typical Usage:
    ```python
    from quanttradeai.data import YFinanceDataSource

    source = YFinanceDataSource()
    df = source.fetch("AAPL", "2024-01-01", "2024-06-01")
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, AsyncIterator
import os
import pandas as pd
import json


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

        # Determine if this is intraday data by checking available columns
        is_intraday = "5. volume" in data.columns

        # Use appropriate column mapping based on data type
        if is_intraday:
            column_mapping = {
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume",
            }
        else:
            column_mapping = {
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "6. volume": "Volume",
            }

        data = data.rename(columns=column_mapping)

        # Fix date filtering for intraday data by converting to datetime if needed
        if is_intraday and isinstance(start, str):
            start = pd.to_datetime(start)
        if is_intraday and isinstance(end, str):
            end = pd.to_datetime(end)

        data = data.loc[start:end]
        return data[["Open", "High", "Low", "Close", "Volume"]]


class WebSocketDataSource(DataSource):
    """Asynchronous data source using a WebSocket streaming API."""

    def __init__(self, url: str) -> None:
        """Initialize the WebSocket data source.

        Parameters
        ----------
        url : str
            WebSocket endpoint provided by the data vendor.
        """
        self.url = url
        self.connection = None

    def fetch(
        self, symbol: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:  # pragma: no cover - not used
        raise NotImplementedError("WebSocketDataSource does not support fetch")

    async def connect(self) -> None:
        """Establish the WebSocket connection."""
        import websockets

        self.connection = await websockets.connect(self.url)

    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time updates for given symbols.

        Parameters
        ----------
        symbols : List[str]
            Ticker symbols to receive updates for.
        """
        if self.connection is None:
            await self.connect()
        message = json.dumps({"type": "subscribe", "symbols": symbols})
        await self.connection.send(message)

    async def stream(self) -> AsyncIterator[dict]:
        """Yield messages from the WebSocket connection."""
        if self.connection is None:
            raise RuntimeError("Connection not established")
        async for message in self.connection:
            yield json.loads(message)

    async def close(self) -> None:
        """Close the WebSocket connection if open."""
        if self.connection is not None:
            await self.connection.close()
