# Data Loading and Processing

API documentation for data fetching, processing, and validation.

## DataLoader Class

### `DataLoader(config_path: str = "config/model_config.yaml", data_source: Optional[DataSource] = None)`

Handles data fetching, caching, and validation for multiple financial instruments.

**Parameters:**
- `config_path` (str): Path to configuration file
- `data_source` (DataSource, optional): Custom data source implementation
- `timeframe` (str, optional): Interval defined in `config/model_config.yaml` (default `'1d'`)

**Example:**
```python
from quanttradeai import DataLoader

# Initialize with default configuration (daily timeframe)
loader = DataLoader("config/model_config.yaml")

# Fetch data for all symbols
data_dict = loader.fetch_data()

# Fetch hourly data for specific symbols
data_dict = loader.fetch_data(symbols=['AAPL', 'META'], refresh=True)
```

### `fetch_data(symbols: Optional[List[str]] = None, refresh: Optional[bool] = None) -> Dict[str, pd.DataFrame]`

Fetches OHLCV data for specified symbols with caching support.

**Parameters:**
- `symbols` (List[str], optional): List of stock symbols. If None, uses symbols from config
- `refresh` (bool, optional): Override cache and fetch fresh data when True

**Returns:**
- `Dict[str, pd.DataFrame]`: Dictionary of DataFrames with OHLCV data for each symbol

**Example:**
```python
# Fetch data for all configured symbols
data = loader.fetch_data()

# Fetch data for specific symbols with cache refresh
data = loader.fetch_data(symbols=['AAPL', 'TSLA'], refresh=True)

# Access data for a specific symbol
aapl_data = data['AAPL']
print(f"AAPL data shape: {aapl_data.shape}")
```

### `validate_data(data_dict: Dict[str, pd.DataFrame]) -> bool`

Validates the fetched data meets requirements.

**Parameters:**
- `data_dict` (Dict[str, pd.DataFrame]): Dictionary of DataFrames with OHLCV data

**Returns:**
- `bool`: True if data is valid, False otherwise

**Example:**
```python
# Validate fetched data
is_valid = loader.validate_data(data_dict)
if not is_valid:
    print("Data validation failed")
```

### `save_data(data_dict: Dict[str, pd.DataFrame], path: Optional[str] = None)`

Saves the fetched data to disk in parquet format.

**Parameters:**
- `data_dict` (Dict[str, pd.DataFrame]): Dictionary of DataFrames to save
- `path` (str, optional): Custom save path

**Example:**
```python
# Save data to default cache directory
loader.save_data(data_dict)

# Save to custom location
loader.save_data(data_dict, "data/custom_cache")
```

### `stream_data(processor, symbols: Optional[List[str]] = None, callback=None)`

Streams real-time data from a `WebSocketDataSource` and processes each update.

**Parameters:**
- `processor` (DataProcessor): Processor instance used to transform incoming data
- `symbols` (List[str], optional): Symbols to subscribe to. Defaults to configured symbols
- `callback` (Callable, optional): Optional function or coroutine invoked with each processed batch

**Example:**
```python
from quanttradeai import DataLoader, DataProcessor, WebSocketDataSource

loader = DataLoader(data_source=WebSocketDataSource("wss://example"))
processor = DataProcessor()

# Print each processed update
async def handle_update(df):
    print(df)

await loader.stream_data(processor, callback=handle_update)
```

## DataSource Classes

### `DataSource` (Abstract Base Class)

Abstract interface for price data providers.

### `YFinanceDataSource`

DataSource implementation using the yfinance package.

**Example:**
```python
from quanttradeai import YFinanceDataSource

# Initialize YFinance data source
data_source = YFinanceDataSource()

# Fetch daily data for a symbol
df = data_source.fetch("AAPL", "2023-01-01", "2023-12-31", interval="1d")
```

### `AlphaVantageDataSource(api_key: Optional[str] = None)`

DataSource implementation for AlphaVantage API.

**Parameters:**
- `api_key` (str, optional): AlphaVantage API key. If None, reads from environment variable

**Example:**
```python
from quanttradeai import AlphaVantageDataSource

# Initialize with API key
data_source = AlphaVantageDataSource("YOUR_API_KEY")

# Fetch hourly data
df = data_source.fetch("AAPL", "2023-01-01", "2023-12-31", interval="1h")
```

### `WebSocketDataSource(url: str)`

Asynchronous data source for streaming market data over WebSocket.

**Parameters:**
- `url` (str): WebSocket endpoint provided by the data vendor

**Example:**
```python
from quanttradeai import WebSocketDataSource

ws_source = WebSocketDataSource("wss://example")
await ws_source.connect()
await ws_source.subscribe(["AAPL"])
async for msg in ws_source.stream():
    print(msg)
```

## DataProcessor Class

### `DataProcessor(config_path: str = "config/features_config.yaml")`

Processes raw OHLCV data and generates required features for the trading strategy.

**Parameters:**
- `config_path` (str): Path to feature configuration file

**Example:**
```python
from quanttradeai import DataProcessor

# Initialize processor
processor = DataProcessor("config/features_config.yaml")

# Process raw data
processed_df = processor.process_data(raw_df)
```

### `process_data(data: pd.DataFrame) -> pd.DataFrame`

Processes raw OHLCV data and generates all required features.

**Parameters:**
- `data` (pd.DataFrame): DataFrame with OHLCV data

**Returns:**
- `pd.DataFrame`: DataFrame with all technical indicators and features

**Example:**
```python
# Process raw OHLCV data
processed_df = processor.process_data(raw_df)

# Check generated features
print(f"Generated {len(processed_df.columns)} features")
print(f"Feature columns: {list(processed_df.columns)}")
```

### `generate_labels(df: pd.DataFrame, forward_returns: int = 5, threshold: float = 0.01) -> pd.DataFrame`

Generates trading signals based on forward returns.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with features
- `forward_returns` (int): Number of days to look ahead
- `threshold` (float): Return threshold for buy/sell signals

**Returns:**
- `pd.DataFrame`: DataFrame with added labels column (1=buy, 0=hold, -1=sell)

**Example:**
```python
# Generate labels for 5-day forward returns with 1% threshold
labeled_df = processor.generate_labels(processed_df, forward_returns=5, threshold=0.01)

# Check label distribution
print(labeled_df['label'].value_counts())
```

## Configuration

### Data Configuration Example
```yaml
data:
  symbols: ['AAPL', 'META', 'TSLA', 'JPM', 'AMZN']
  start_date: '2015-01-01'
  end_date: '2024-12-31'
  timeframe: '1d'
  cache_dir: 'data/raw'
  cache_expiration_days: 7
  use_cache: true
  refresh: false
  max_workers: 1
```

### Feature Configuration Example
```yaml
price_features:
  sma_periods: [5, 10, 20, 50, 200]
  ema_periods: [5, 10, 20, 50, 200]

momentum_features:
  rsi_period: 14
  macd_params:
    fast: 12
    slow: 26
    signal: 9

preprocessing:
  scaling:
    method: 'standard'
  outliers:
    method: 'winsorize'
    limits: [0.01, 0.99]
```

## Error Handling

### Common Data Issues
```python
# Check cache directory
import os
print(os.path.exists("data/raw"))

# Force refresh data
data = loader.fetch_data(refresh=True)

# Check for missing values
print(df.isnull().sum())
df = df.fillna(method='ffill')
```

### Validation Errors
```python
try:
    # Validate data
    is_valid = loader.validate_data(data_dict)
    if not is_valid:
        print("Data validation failed")
except Exception as e:
    print(f"Validation error: {e}")
```

## Related Documentation

- **[Feature Engineering](features.md)** - Technical indicators and custom features
- **[Machine Learning](models.md)** - Model training and evaluation
- **[Configuration](../configuration.md)** - Configuration guide
- **[Quick Reference](../quick-reference.md)** - Common patterns