# Data API

## Overview

The data API loads OHLCV bars, validates raw market data, attaches optional news text, and creates model-ready feature frames. The top-level imports below are lazy exports from `quanttradeai.__init__`; direct subpackage imports are also valid.

## Public Imports

```python
from quanttradeai import (
    DataSource,
    YFinanceDataSource,
    AlphaVantageDataSource,
    WebSocketDataSource,
    DataLoader,
    DataProcessor,
)

from quanttradeai.data import DataLoader, DataProcessor
from quanttradeai.data.datasource import DataSource, YFinanceDataSource
```

## Main Classes

| API | Import Path | Purpose |
| --- | --- | --- |
| `DataSource` | `quanttradeai.data.datasource` | Abstract OHLCV provider interface |
| `YFinanceDataSource` | `quanttradeai.data.datasource` | Historical Yahoo Finance OHLCV provider |
| `AlphaVantageDataSource` | `quanttradeai.data.datasource` | Alpha Vantage daily/intraday OHLCV provider |
| `WebSocketDataSource` | `quanttradeai.data.datasource` | Generic async WebSocket message source |
| `DataLoader` | `quanttradeai.data.loader` | Config-driven fetch, cache, validation, and save orchestration |
| `DataProcessor` | `quanttradeai.data.processor` | Feature generation, preprocessing, and label generation |

## `DataSource`

**Signature**

```python
class DataSource(ABC):
    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame: ...
```

`DataSource` is the abstract interface used by `DataLoader`. Implementations must return a `pandas.DataFrame` indexed by datetime-like values and should include standard OHLCV columns when the data is intended for processing:

| Column | Required By |
| --- | --- |
| `Open` | validation, technical features |
| `High` | validation, stochastic, ATR, volatility features |
| `Low` | validation, stochastic, ATR, volatility features |
| `Close` | validation, labels, most indicators |
| `Volume` | validation, volume features, liquidity simulation |

**Minimal custom provider**

```python
import pandas as pd
from quanttradeai import DataSource


class CsvDataSource(DataSource):
    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        df = pd.read_csv(f"data/raw/{symbol}.csv", parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        return df.loc[start:end]
```

## `YFinanceDataSource`

**Signature**

```python
class YFinanceDataSource(DataSource):
    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame
```

Uses `yfinance.Ticker(symbol).history(...)` and returns Yahoo Finance's DataFrame directly.

**Supported intervals**

Common Yahoo intervals are supported, including `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, and `3mo`.

```python
from quanttradeai import YFinanceDataSource

source = YFinanceDataSource()
bars = source.fetch("AAPL", "2024-01-01", "2024-06-01", interval="1d")
```

**Edge cases**

- Empty vendor responses are returned as empty DataFrames; `DataLoader` logs the symbol and omits it from the returned mapping.
- Column availability follows `yfinance`; adjusted or corporate-action columns may appear in addition to OHLCV.

## `AlphaVantageDataSource`

**Signature**

```python
class AlphaVantageDataSource(DataSource):
    def __init__(self, api_key: Optional[str] = None) -> None
    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame
```

Uses `alpha_vantage.timeseries.TimeSeries` and normalizes vendor columns to `Open`, `High`, `Low`, `Close`, and `Volume`.

| Parameter | Meaning |
| --- | --- |
| `api_key` | Explicit Alpha Vantage API key. If omitted, `ALPHAVANTAGE_API_KEY` is used. |
| `interval` | `1d`/`daily`, or intraday `1min`, `5min`, `15min`, `30min`, `60min`, `1h`. |

```python
from quanttradeai import AlphaVantageDataSource

source = AlphaVantageDataSource()
bars = source.fetch("MSFT", "2024-01-01", "2024-03-01", interval="1d")
```

**Errors and edge cases**

- Raises `ValueError` during construction if no API key is provided and `ALPHAVANTAGE_API_KEY` is unset.
- Raises `ValueError` for unsupported intraday intervals.
- Date filtering is applied after the full vendor response is loaded.

## `WebSocketDataSource`

**Signature**

```python
class WebSocketDataSource(DataSource):
    def __init__(self, url: str) -> None
    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame
    async def connect(self) -> None
    async def subscribe(self, symbols: list[str]) -> None
    async def stream(self) -> AsyncIterator[dict]
    async def close(self) -> None
```

`WebSocketDataSource` is a generic async source for JSON messages. It does not support historical `fetch`; calling `fetch` raises `NotImplementedError`.

```python
from quanttradeai import WebSocketDataSource

source = WebSocketDataSource("wss://example.test/market-data")
await source.subscribe(["AAPL", "MSFT"])

async for message in source.stream():
    print(message)
    break

await source.close()
```

**Errors and edge cases**

- `stream()` raises `RuntimeError` if called before a connection exists.
- `subscribe()` auto-connects if needed and sends `{"type": "subscribe", "symbols": symbols}` as JSON.

## `DataLoader`

**Signature**

```python
class DataLoader:
    def __init__(
        self,
        config_path: str = "config/model_config.yaml",
        data_source: Optional[DataSource] = None,
        news_data_source: Optional[NewsDataSource] = None,
    )

    def fetch_data(
        self,
        symbols: Optional[list[str]] = None,
        refresh: Optional[bool] = None,
    ) -> dict[str, pd.DataFrame]

    def validate_data(self, data_dict: dict[str, pd.DataFrame]) -> tuple[bool, dict]
    def save_data(self, data_dict: dict[str, pd.DataFrame], path: Optional[str] = None) -> None
    async def stream_data(self, processor, symbols: Optional[list[str]] = None, callback=None) -> None
```

`DataLoader` reads a runtime model config file validated by `ModelConfigSchema`. It fetches each configured symbol, optionally caches parquet files, validates missing dates, joins secondary timeframes, and can attach news text when the runtime config enables news.

| Method | Returns | Notes |
| --- | --- | --- |
| `fetch_data(...)` | `dict[str, pd.DataFrame]` | Omits symbols that fail or return no data. |
| `validate_data(...)` | `(bool, dict)` | Checks OHLCV columns, date span, and NaN ratios. |
| `save_data(...)` | `None` | Writes `{SYMBOL}_data.parquet` files. |
| `stream_data(...)` | `None` | Requires `data_source` to be `WebSocketDataSource`; calls `processor.process_data(...)`. |

```python
from quanttradeai import DataLoader, YFinanceDataSource

loader = DataLoader(
    config_path="config/model_config.yaml",
    data_source=YFinanceDataSource(),
)

frames = loader.fetch_data(symbols=["AAPL"], refresh=True)
passed, report = loader.validate_data(frames)
```

**Important behavior**

- Constructor raises `FileNotFoundError` if `config_path` is missing.
- Constructor raises `ValueError` if the config fails schema validation.
- Cache files are named `{symbol}_{timeframe}_data.parquet`.
- `max_workers > 1` fetches symbols concurrently through a thread pool.
- `validate_data` requires at least one year of date span and a maximum OHLCV NaN ratio of `0.01`.

## `DataProcessor`

**Signature**

```python
class DataProcessor:
    def __init__(self, config_path: str = "config/features_config.yaml")
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame
    def create_preprocessor(self) -> FeaturePreprocessor
    def generate_labels(
        self,
        df: pd.DataFrame,
        forward_returns: int = 5,
        threshold: float = 0.01,
    ) -> pd.DataFrame
```

`DataProcessor` turns OHLCV frames into feature frames. It reads a runtime features config when present; if the file is missing it uses built-in defaults.

| Method | Purpose | Return |
| --- | --- | --- |
| `generate_features(data)` | Adds causal technical, volume, custom, multi-timeframe, and optional sentiment features. | Feature DataFrame |
| `create_preprocessor()` | Builds a `FeaturePreprocessor` from configured scaling/outlier/selection settings. | Preprocessor object |
| `process_data(data)` | Convenience path that generates features and fits/transforms preprocessing on the same input. | Processed DataFrame |
| `generate_labels(df, forward_returns=5, threshold=0.01)` | Adds `forward_returns` and `label`. | Labeled DataFrame |

```python
from quanttradeai import DataProcessor

processor = DataProcessor("config/features_config.yaml")
features = processor.generate_features(bars)
labeled = processor.generate_labels(features, forward_returns=5, threshold=0.01)
```

For train/test work, prefer fitting preprocessing on the train slice and transforming the test slice:

```python
features = processor.generate_features(bars)
train = features.loc[: "2024-06-30"]
test = features.loc["2024-07-01" :]

preprocessor = processor.create_preprocessor().fit(train)
train_ready = preprocessor.transform(train)
test_ready = preprocessor.transform(test)
```

**Expected columns**

At minimum, `DataProcessor` expects `Open`, `High`, `Low`, `Close`, and `Volume` for the default indicator pipeline. Sentiment scoring also expects a `text` column when sentiment is enabled.

**Important behavior**

- `generate_features` drops the first 200 rows after feature generation, forward-fills remaining values, and then drops any remaining NaNs.
- `generate_labels` creates labels: `1` for forward returns above `threshold`, `-1` below `-threshold`, and `0` otherwise.
- Missing feature config files fall back to built-in defaults.
- Schema validation errors raise `ValueError`; YAML parsing issues are logged and fall back to defaults.
- Sentiment features require a configured `SentimentAnalyzer` and the referenced API key environment variable.

## Minimal Examples

### Fetch, Process, Label

```python
from quanttradeai import DataLoader, DataProcessor

loader = DataLoader("config/model_config.yaml")
processor = DataProcessor("config/features_config.yaml")

frames = loader.fetch_data(["AAPL"])
features = processor.generate_features(frames["AAPL"])
labeled = processor.generate_labels(features)
```

### Use a Custom Provider

```python
loader = DataLoader(
    "config/model_config.yaml",
    data_source=CsvDataSource(),
)
frames = loader.fetch_data(["AAPL"])
```

## Related CLI/YAML Docs

- [CLI docs](../cli/)
- [Config docs](../config/)
- [Examples](../examples/)

The CLI research path compiles `config/project.yaml` into the runtime model and feature configs consumed by `DataLoader` and `DataProcessor`.

## Common Mistakes

- Passing lowercase `open/high/low/close/volume` columns into `DataProcessor`; the default pipeline expects title-case OHLCV columns.
- Calling `WebSocketDataSource.fetch(...)`; streaming sources expose async `subscribe` and `stream`, not historical fetch.
- Treating empty fetch results as exceptions; `DataLoader.fetch_data` logs and skips failed symbols.
- Fitting preprocessing on the full dataset before time-based evaluation.
- Forgetting that `generate_features` removes the first 200 rows for indicator warm-up.
