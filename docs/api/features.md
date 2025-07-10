# Feature Engineering

API documentation for technical indicators and custom features.

## Technical Indicators

### `sma(series: pd.Series, period: int) -> pd.Series`

Calculates Simple Moving Average.

**Parameters:**
- `series` (pd.Series): Price series
- `period` (int): Moving average period

**Returns:**
- `pd.Series`: Simple moving average

**Example:**
```python
from src.features.technical import sma

# Calculate 20-period SMA
sma_20 = sma(df['Close'], 20)
```

### `ema(series: pd.Series, period: int) -> pd.Series`

Calculates Exponential Moving Average.

**Parameters:**
- `series` (pd.Series): Price series
- `period` (int): Moving average period

**Returns:**
- `pd.Series`: Exponential moving average

**Example:**
```python
from src.features.technical import ema

# Calculate 20-period EMA
ema_20 = ema(df['Close'], 20)
```

### `rsi(series: pd.Series, period: int = 14) -> pd.Series`

Calculates Relative Strength Index.

**Parameters:**
- `series` (pd.Series): Price series
- `period` (int): RSI period (default: 14)

**Returns:**
- `pd.Series`: RSI values

**Example:**
```python
from src.features.technical import rsi

# Calculate 14-period RSI
rsi_14 = rsi(df['Close'], 14)
```

### `macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame`

Calculates MACD indicator.

**Parameters:**
- `series` (pd.Series): Price series
- `fast` (int): Fast EMA period (default: 12)
- `slow` (int): Slow EMA period (default: 26)
- `signal` (int): Signal line period (default: 9)

**Returns:**
- `pd.DataFrame`: DataFrame with 'macd', 'signal', and 'hist' columns

**Example:**
```python
from src.features.technical import macd

# Calculate MACD
macd_df = macd(df['Close'])
macd_line = macd_df['macd']
signal_line = macd_df['signal']
histogram = macd_df['hist']
```

### `stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame`

Calculates Stochastic Oscillator.

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `k` (int): %K period (default: 14)
- `d` (int): %D period (default: 3)

**Returns:**
- `pd.DataFrame`: DataFrame with 'stoch_k' and 'stoch_d' columns

**Example:**
```python
from src.features.technical import stochastic

# Calculate Stochastic Oscillator
stoch_df = stochastic(df['High'], df['Low'], df['Close'])
stoch_k = stoch_df['stoch_k']
stoch_d = stoch_df['stoch_d']
```

## Custom Features

### `momentum_score(close: pd.Series, sma: pd.Series, rsi_series: pd.Series, macd: pd.Series, macd_signal: pd.Series) -> pd.Series`

Computes a simple momentum score from multiple indicators.

**Parameters:**
- `close` (pd.Series): Close prices
- `sma` (pd.Series): Simple moving average
- `rsi_series` (pd.Series): RSI values
- `macd` (pd.Series): MACD line
- `macd_signal` (pd.Series): MACD signal line

**Returns:**
- `pd.Series`: Normalized momentum score

**Example:**
```python
from src.features.custom import momentum_score

# Calculate momentum score
score = momentum_score(df['Close'], df['sma_20'], df['rsi'], df['macd'], df['macd_signal'])
```

### `volatility_breakout(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 20, threshold: float = 2.0) -> pd.Series`

Flags days when price breaks above the previous high plus a threshold.

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `lookback` (int): Lookback period (default: 20)
- `threshold` (float): Breakout threshold (default: 2.0)

**Returns:**
- `pd.Series`: Binary breakout signals

**Example:**
```python
from src.features.custom import volatility_breakout

# Calculate volatility breakout signals
breakout = volatility_breakout(df['High'], df['Low'], df['Close'])
```

## Feature Generation Patterns

### Moving Averages
```python
from src.features.technical import sma, ema

# Generate multiple moving averages
periods = [5, 10, 20, 50, 200]
for period in periods:
    df[f'sma_{period}'] = sma(df['Close'], period)
    df[f'ema_{period}'] = ema(df['Close'], period)
```

### Momentum Indicators
```python
from src.features.technical import rsi, macd, stochastic

# Generate momentum indicators
df['rsi'] = rsi(df['Close'], 14)
macd_df = macd(df['Close'])
df['macd'] = macd_df['macd']
df['macd_signal'] = macd_df['signal']

stoch_df = stochastic(df['High'], df['Low'], df['Close'])
df['stoch_k'] = stoch_df['stoch_k']
df['stoch_d'] = stoch_df['stoch_d']
```

### Custom Features
```python
from src.features.custom import momentum_score, volatility_breakout

# Generate custom features
df['momentum_score'] = momentum_score(
    df['Close'], 
    df['sma_20'], 
    df['rsi'], 
    df['macd'], 
    df['macd_signal']
)

df['volatility_breakout'] = volatility_breakout(
    df['High'], 
    df['Low'], 
    df['Close']
)
```

## Configuration

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
  stoch_params:
    k: 14
    d: 3

volatility_features:
  bollinger_bands:
    period: 20
    std_dev: 2

volume_features:
  volume_sma:
    periods: [5, 10, 20]
  volume_ema:
    periods: [5, 10, 20]

feature_combinations:
  cross_indicators:
    - ['sma_5', 'sma_20']
    - ['ema_5', 'ema_20']
    - ['close', 'sma_50']
  ratio_indicators:
    - ['volume', 'volume_sma_5']
    - ['close', 'sma_20']
    - ['high', 'low']
```

## Error Handling

### Missing Data
```python
# Check for NaN values in indicators
print(df[['sma_20', 'rsi', 'macd']].isnull().sum())

# Handle missing values
df = df.fillna(method='ffill')
```

### Invalid Parameters
```python
try:
    # Calculate RSI with valid period
    rsi_14 = rsi(df['Close'], 14)
except Exception as e:
    print(f"Error calculating RSI: {e}")
```

## Performance Tips

### Vectorized Operations
```python
# Use vectorized operations for better performance
df['price_change'] = df['Close'].pct_change()
df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
```

### Memory Optimization
```python
# Drop unnecessary columns to save memory
columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'sma_20', 'rsi', 'macd']
df = df[columns_to_keep]
```

## Related Documentation

- **[Data Loading](data.md)** - Data fetching and processing
- **[Machine Learning](models.md)** - Model training and evaluation
- **[Configuration](../configuration.md)** - Configuration guide
- **[Quick Reference](../quick-reference.md)** - Common patterns