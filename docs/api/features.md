# Features API

## Overview

Feature helpers live under `quanttradeai.features`. The package exports modules, not individual functions, so import the module namespace you need. `DataProcessor` uses these helpers to build configured feature pipelines.

## Public Imports

```python
from quanttradeai.features import technical, custom, sentiment

from quanttradeai.features.technical import sma, ema, rsi, macd, stochastic
from quanttradeai.features.custom import momentum_score, volatility_breakout
from quanttradeai.features.sentiment import SentimentAnalyzer
```

## Main Classes and Functions

| API | Import Path | Purpose |
| --- | --- | --- |
| `sma(series, period)` | `quanttradeai.features.technical` | Simple moving average |
| `ema(series, period)` | `quanttradeai.features.technical` | Exponential moving average |
| `rsi(series, period=14)` | `quanttradeai.features.technical` | Relative Strength Index |
| `macd(series, fast=12, slow=26, signal=9)` | `quanttradeai.features.technical` | MACD, signal, histogram DataFrame |
| `stochastic(high, low, close, k=14, d=3)` | `quanttradeai.features.technical` | Stochastic oscillator DataFrame |
| `momentum_score(close, sma, rsi_series, macd, macd_signal)` | `quanttradeai.features.custom` | Weighted normalized momentum score |
| `volatility_breakout(high, low, close, lookback=20, threshold=2.0)` | `quanttradeai.features.custom` | Binary breakout flag |
| `SentimentAnalyzer(provider, model, api_key_env_var, extra=None)` | `quanttradeai.features.sentiment` | LiteLLM-backed sentiment scorer |

## Technical Indicators

### `sma(series: pd.Series, period: int) -> pd.Series`

Returns a simple moving average from `pandas_ta_classic.sma`.

```python
from quanttradeai.features.technical import sma

df["sma_20"] = sma(df["Close"], 20)
```

### `ema(series: pd.Series, period: int) -> pd.Series`

Returns an exponential moving average from `pandas_ta_classic.ema`.

```python
from quanttradeai.features.technical import ema

df["ema_20"] = ema(df["Close"], 20)
```

### `rsi(series: pd.Series, period: int = 14) -> pd.Series`

Returns RSI values from `pandas_ta_classic.rsi`.

```python
from quanttradeai.features.technical import rsi

df["rsi"] = rsi(df["Close"], period=14)
```

### `macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame`

Returns a DataFrame with stable column names:

| Column | Meaning |
| --- | --- |
| `macd` | MACD line |
| `signal` | Signal line |
| `hist` | Histogram |

```python
from quanttradeai.features.technical import macd

macd_frame = macd(df["Close"])
df["macd"] = macd_frame["macd"]
df["macd_signal"] = macd_frame["signal"]
df["macd_hist"] = macd_frame["hist"]
```

### `stochastic(high, low, close, k: int = 14, d: int = 3) -> pd.DataFrame`

Returns a DataFrame with stable column names:

| Column | Meaning |
| --- | --- |
| `stoch_k` | Stochastic %K |
| `stoch_d` | Stochastic %D |

```python
from quanttradeai.features.technical import stochastic

stoch = stochastic(df["High"], df["Low"], df["Close"], k=14, d=3)
df = df.join(stoch)
```

**Expected inputs**

All technical functions expect numeric `pandas.Series` inputs. Insufficient lookback history produces NaNs, which `DataProcessor.generate_features` later cleans after indicator warm-up.

## Custom Features

### `momentum_score(...) -> pd.Series`

**Signature**

```python
def momentum_score(
    close: pd.Series,
    sma: pd.Series,
    rsi_series: pd.Series,
    macd: pd.Series,
    macd_signal: pd.Series,
) -> pd.Series
```

Computes a weighted score:

- `close > sma`: weight `0.3`
- `rsi_series > 50`: weight `0.3`
- `macd > macd_signal`: weight `0.4`

The score is then standardized by subtracting its mean and dividing by its standard deviation.

```python
from quanttradeai.features.custom import momentum_score

df["momentum_score"] = momentum_score(
    df["Close"],
    df["sma_20"],
    df["rsi"],
    df["macd"],
    df["macd_signal"],
)
```

**Edge case**

If the underlying weighted score has zero standard deviation, the normalized result can contain NaNs.

### `volatility_breakout(...) -> pd.Series`

**Signature**

```python
def volatility_breakout(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 20,
    threshold: float = 2.0,
) -> pd.Series
```

Flags rows where `close` breaks above the previous rolling high plus `threshold * (rolling_high - rolling_low)`.

```python
from quanttradeai.features.custom import volatility_breakout

df["volatility_breakout_20"] = volatility_breakout(
    df["High"],
    df["Low"],
    df["Close"],
    lookback=20,
    threshold=2.0,
)
```

Returns a binary integer Series with `1` for breakout rows and `0` otherwise.

## Sentiment

### `SentimentAnalyzer`

**Signature**

```python
class SentimentAnalyzer:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key_env_var: str,
        extra: dict[str, Any] | None = None,
    ) -> None

    def score(self, text: str) -> float
```

`SentimentAnalyzer` uses LiteLLM to ask a configured provider for a numeric sentiment score between `-1` and `1`.

```python
from quanttradeai.features.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer(
    provider="openai",
    model="provider-model",
    api_key_env_var="OPENAI_API_KEY",
)

score = analyzer.score("Earnings beat expectations and guidance improved.")
```

**Errors and edge cases**

- Raises `ValueError` if `provider` or `model` is blank.
- Raises `ValueError` if `api_key_env_var` is not set in the environment.
- Raises `ValueError` if the model response cannot be parsed as a float.
- Network/provider errors from LiteLLM are re-raised.

## `DataProcessor` Feature Pipeline

`quanttradeai.data.processor.DataProcessor` composes the feature helpers above. Depending on config, it can add:

| Group | Generated Columns |
| --- | --- |
| Price ratios | `close_to_open`, `high_to_low`, `close_to_high`, `close_to_low`, `price_range` |
| Momentum | `sma_*`, `ema_*`, `rsi`, `macd`, `macd_signal`, `macd_hist`, `stoch_k`, `stoch_d` |
| Volatility | `bb_lower`, `bb_middle`, `bb_upper`, `atr_*`, `keltner_*` |
| Returns | `daily_return`, `weekly_return`, `monthly_return`, `volatility_21d` |
| Volume | `volume_sma_*`, `volume_ema_*`, volume ratios, `obv`, `volume_price_trend` |
| Custom | `price_momentum_*`, `volume_momentum_*`, `mean_reversion_*`, `volatility_breakout_*`, `momentum_score` |
| Sentiment | `sentiment_score` when enabled and `text` exists |

```python
from quanttradeai import DataProcessor

processor = DataProcessor("config/features_config.yaml")
features = processor.generate_features(df)
```

## Minimal Examples

### Manual Indicator Set

```python
from quanttradeai.features import technical

features = df.copy()
features["sma_20"] = technical.sma(features["Close"], 20)
features["rsi"] = technical.rsi(features["Close"], 14)
features = features.join(technical.macd(features["Close"]))
```

### Configured Feature Generation

```python
from quanttradeai import DataProcessor

processor = DataProcessor("config/features_config.yaml")
features = processor.generate_features(df)
```

## Related CLI/YAML Docs

- [CLI docs](../cli/)
- [Config docs](../config/)
- [Examples](../examples/)

The CLI research workflow uses the same `DataProcessor` feature generation path after compiling project YAML into runtime feature settings.

## Common Mistakes

- Importing `sma` from `quanttradeai.features` directly; import from `quanttradeai.features.technical`.
- Passing DataFrames where helper functions expect Series.
- Forgetting indicator warm-up rows; rolling indicators produce NaNs until enough history exists.
- Enabling sentiment without a `text` column or without the configured provider key environment variable.
- Recomputing feature preprocessing independently for training and serving instead of reusing a fitted preprocessor.
