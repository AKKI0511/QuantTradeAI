# Configuration Guide

Learn how to configure QuantTradeAI for your specific needs.

## 📁 Configuration Files

The framework uses two main configuration files:

- **`config/model_config.yaml`** - Model parameters and data settings
- **`config/features_config.yaml`** - Feature engineering settings

LiteLLM is bundled for sentiment analysis. To install manually:

```bash
poetry add litellm  # or pip install litellm
```

## 🔧 Model Configuration

### Data Settings

```yaml
data:
  symbols: ['AAPL', 'META', 'TSLA', 'JPM', 'AMZN']
  start_date: '2015-01-01'
  end_date: '2024-12-31'
  cache_dir: 'data/raw'
  cache_path: 'data/raw'
  cache_expiration_days: 7
  use_cache: true
  refresh: false
  max_workers: 1
  test_start: '2025-01-01'
  test_end: '2025-01-31'
```

**Key Parameters:**
- `symbols`: List of stock symbols to process
- `start_date`/`end_date`: Data date range
- `cache_dir`: Directory for cached data
- `use_cache`: Enable/disable caching
- `refresh`: Force fresh data download
- `max_workers`: Parallel processing workers

### Model Parameters

```yaml
models:
  voting_classifier:
    voting: 'soft'
    weights: [1, 2, 2]
  
  logistic_regression:
    C: 1.0
    max_iter: 1000
    class_weight: 'balanced'
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    class_weight: 'balanced'
  
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
```

### Training Settings

```yaml
training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
```

### Trading Parameters

```yaml
trading:
  position_size: 0.2
  stop_loss: 0.02
  take_profit: 0.04
  max_positions: 5
  transaction_cost: 0.001
```

## 🔧 Feature Configuration

### Price Features

```yaml
price_features:
  sma_periods: [5, 10, 20, 50, 200]
  ema_periods: [5, 10, 20, 50, 200]
```

### Momentum Features

```yaml
momentum_features:
  rsi_period: 14
  macd_params:
    fast: 12
    slow: 26
    signal: 9
  stoch_params:
    k: 14
    d: 3
```

### Volatility Features

```yaml
volatility_features:
  bollinger_bands:
    period: 20
    std_dev: 2
```

### Volume Features

```yaml
volume_features:
  volume_sma:
    periods: [5, 10, 20]
  volume_ema:
    periods: [5, 10, 20]
```

### Feature Combinations

```yaml
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

### Sentiment Features

```yaml
sentiment:
  enabled: true
  provider: openai  # e.g. openai, anthropic, huggingface, ollama
  model: gpt-3.5-turbo
  api_key_env_var: OPENAI_API_KEY
  extra: {}
```

Set the API key before running:

```bash
export OPENAI_API_KEY="sk-..."
```

### Preprocessing

```yaml
preprocessing:
  scaling:
    method: 'standard'  # Options: standard, minmax, robust
    target_range: [-1, 1]
  
  outliers:
    method: 'winsorize'  # Options: winsorize, clip
    limits: [0.01, 0.99]
```

### Feature Selection

```yaml
feature_selection:
  method: 'recursive'  # Options: recursive, lasso, random_forest
  n_features: 20
  scoring: 'f1'
```

### Pipeline Steps

```yaml
pipeline:
  steps:
    - generate_technical_indicators
    - generate_volume_features
    - generate_custom_features
    - generate_sentiment
    - handle_missing_values
    - remove_outliers
    - scale_features
    - select_features
```

## 🎯 Common Configurations

### Minimal Configuration
```yaml
data:
  symbols: ['AAPL']
  start_date: '2023-01-01'
  end_date: '2024-12-31'
  use_cache: true

models:
  voting_classifier:
    voting: 'soft'
```

### Production Configuration
```yaml
data:
  symbols: ['AAPL', 'META', 'TSLA', 'JPM', 'AMZN']
  start_date: '2015-01-01'
  end_date: '2024-12-31'
  cache_dir: 'data/raw'
  use_cache: true
  max_workers: 4

models:
  voting_classifier:
    voting: 'soft'
    weights: [1, 2, 2]
  
  xgboost:
    n_estimators: 200
    max_depth: 8
    learning_rate: 0.05
```

### Research Configuration
```yaml
data:
  symbols: ['AAPL', 'TSLA']
  start_date: '2020-01-01'
  end_date: '2024-12-31'
  refresh: true

features:
  price_features:
    sma_periods: [5, 10, 20, 50]
    ema_periods: [5, 10, 20, 50]
  
  momentum_features:
    rsi_period: 14
    macd_params:
      fast: 12
      slow: 26
      signal: 9
```

## 🔍 Configuration Validation

The framework validates configuration files using Pydantic schemas:

```python
from quanttradeai.utils.config_schemas import ModelConfigSchema, FeaturesConfigSchema
import yaml

# Validate model config
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    ModelConfigSchema(**config)

# Validate features config
with open("config/features_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    FeaturesConfigSchema(**config)
```

## 🚨 Common Issues

### Invalid Configuration
```yaml
# ❌ Wrong
data:
  symbols: 'AAPL'  # Should be list

# ✅ Correct
data:
  symbols: ['AAPL']
```

### Missing Required Fields
```yaml
# ❌ Missing required field
data:
  symbols: ['AAPL']
  # Missing start_date and end_date

# ✅ Complete
data:
  symbols: ['AAPL']
  start_date: '2023-01-01'
  end_date: '2024-12-31'
```

### Invalid Date Format
```yaml
# ❌ Wrong format
data:
  start_date: '2023/01/01'  # Use YYYY-MM-DD

# ✅ Correct format
data:
  start_date: '2023-01-01'
```

## 📚 Related Documentation

- **[Getting Started](getting-started.md)** - Installation and first steps
- **[Quick Reference](quick-reference.md)** - Common patterns and commands
- **[API Reference](api/)** - Complete API documentation
