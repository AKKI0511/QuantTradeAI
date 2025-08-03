"""Historical market data loader.

The :class:`DataLoader` class retrieves, validates and caches OHLCV data
from a configurable :class:`~quanttradeai.data.datasource.DataSource`.

Key Components:
    - :class:`DataLoader`: orchestrates data downloading and caching

Typical Usage:
    ```python
    from quanttradeai.data import DataLoader
    loader = DataLoader("config/model_config.yaml")
    frames = loader.fetch_data()
    ```
"""

import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import yaml
from pydantic import ValidationError
from concurrent.futures import ThreadPoolExecutor, as_completed

from quanttradeai.utils.config_schemas import ModelConfigSchema
from quanttradeai.data.datasource import DataSource, YFinanceDataSource
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader class for fetching and validating stock data."""

    def __init__(
        self,
        config_path: str = "config/model_config.yaml",
        data_source: Optional[DataSource] = None,
    ):
        """Initialize DataLoader with configuration and validate it."""
        with open(config_path, "r") as file:
            raw_cfg = yaml.safe_load(file)

        try:
            schema = ModelConfigSchema(**raw_cfg)
        except ValidationError as exc:
            raise ValueError(f"Invalid model configuration: {exc}") from exc

        data_cfg = schema.data
        self.config = raw_cfg
        self.symbols = data_cfg.symbols
        self.start_date = data_cfg.start_date
        self.end_date = data_cfg.end_date
        self.timeframe = data_cfg.timeframe or "1d"
        # allow both legacy 'cache_dir' and new 'cache_path' keys
        self.cache_dir = data_cfg.cache_path or data_cfg.cache_dir or "data/raw"
        self.cache_expiration_days = data_cfg.cache_expiration_days
        self.use_cache = data_cfg.use_cache
        self.default_refresh = data_cfg.refresh
        self.max_workers = data_cfg.max_workers or 1
        self.data_source = data_source or YFinanceDataSource()

    def _is_cache_valid(self, cache_file: str) -> bool:
        """Return True if the cache file exists and is not expired."""
        if not os.path.exists(cache_file):
            return False
        if self.cache_expiration_days is None:
            return True
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return datetime.now() - file_time < timedelta(days=self.cache_expiration_days)

    def _fetch_single(self, symbol: str, refresh: bool) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol and handle caching."""
        cache_file = os.path.join(
            self.cache_dir, f"{symbol}_{self.timeframe}_data.parquet"
        )
        try:
            if self.use_cache and not refresh and self._is_cache_valid(cache_file):
                logger.info(f"Loading cached data for {symbol} from {cache_file}")
                df = pd.read_parquet(cache_file)
            else:
                logger.info(f"Fetching data for {symbol}")
                df = self.data_source.fetch(
                    symbol, self.start_date, self.end_date, self.timeframe
                )

                if df is None or df.empty:
                    logger.error(f"No data found for {symbol}")
                    return None

                if self.use_cache:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    df.to_parquet(cache_file)
                    logger.info(f"Cached data for {symbol} at {cache_file}")

            missing_dates = self._check_missing_dates(df)
            if missing_dates:
                logger.warning(f"Missing dates for {symbol}: {len(missing_dates)} days")

            logger.info(f"Successfully retrieved {len(df)} records for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def fetch_data(
        self, symbols: Optional[List[str]] = None, refresh: Optional[bool] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for specified symbols.

        Args:
            symbols: List of stock symbols. If None, uses symbols from config.
            refresh: Override cache and fetch fresh data when True.

        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol.
        """
        symbols = symbols or self.symbols
        refresh = self.default_refresh if refresh is None else refresh
        data_dict: Dict[str, pd.DataFrame] = {}

        if self.max_workers and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._fetch_single, s, refresh): s for s in symbols
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    df = future.result()
                    if df is not None:
                        data_dict[symbol] = df
        else:
            for symbol in symbols:
                df = self._fetch_single(symbol, refresh)
                if df is not None:
                    data_dict[symbol] = df

        return data_dict

    def _check_missing_dates(self, df: pd.DataFrame) -> List[datetime]:
        """Check for missing trading days in the data."""
        all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
        missing_dates = all_dates.difference(df.index)
        return list(missing_dates)

    def validate_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate the fetched data meets requirements.

        Args:
            data_dict: Dictionary of DataFrames with OHLCV data.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        for symbol, df in data_dict.items():
            # Check required columns
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns for {symbol}")
                return False

            # Check data range
            date_range = (df.index.max() - df.index.min()).days
            if date_range < 365:  # At least one year of data
                logger.error(f"Insufficient data range for {symbol}")
                return False

            # Check for excessive missing values
            # Check missing value ratio per column
            if df.isnull().mean().max() > 0.01:  # Max 1% missing values
                logger.error(f"Too many missing values for {symbol}")
                return False

        return True

    def save_data(
        self, data_dict: Dict[str, pd.DataFrame], path: Optional[str] = None
    ) -> None:
        """Save the fetched data to disk."""
        import os

        path = path or self.cache_dir
        os.makedirs(path, exist_ok=True)

        for symbol, df in data_dict.items():
            file_path = f"{path}/{symbol}_data.parquet"
            df.to_parquet(file_path)
            logger.info(f"Saved data for {symbol} to {file_path}")

    async def stream_data(
        self,
        processor,
        symbols: Optional[List[str]] = None,
        callback=None,
    ) -> None:
        """Stream real-time data and dispatch to the processing pipeline.

        Parameters
        ----------
        processor : DataProcessor
            Processor instance used to transform incoming data.
        symbols : Optional[List[str]], optional
            Symbols to subscribe to. Defaults to the loader's configured symbols.
        callback : Callable, optional
            Optional function or coroutine invoked with each processed batch.
        """

        from quanttradeai.data.datasource import WebSocketDataSource
        import asyncio

        if not isinstance(self.data_source, WebSocketDataSource):
            raise TypeError("Streaming requires WebSocketDataSource")

        symbols = symbols or self.symbols
        await self.data_source.subscribe(symbols)

        async for msg in self.data_source.stream():
            df = pd.DataFrame([msg])
            processed = processor.process_data(df)
            if callback:
                res = callback(processed)
                if asyncio.iscoroutine(res):
                    await res
