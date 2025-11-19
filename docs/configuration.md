# Configuration Guide

Learn how to configure QuantTradeAI for your specific needs.

## 📁 Configuration Files

The framework uses several configuration files:

- **`config/model_config.yaml`** - Model parameters and data settings
- **`config/features_config.yaml`** - Feature engineering settings
- **`config/backtest_config.yaml`** - Execution settings for backtests
- **`config/impact_config.yaml`** - Market impact parameters by asset class
- **`config/risk_config.yaml`** - Drawdown protection and turnover limits
- **`config/position_manager.yaml`** - Live position tracking and intraday risk controls

## 🔧 Model Configuration

### Data Settings

```yaml
data:
  symbols: ['AAPL', 'META', 'TSLA', 'JPM', 'AMZN']
  start_date: '2015-01-01'
  end_date: '2024-12-31'
  cache_dir: 'data/raw'
  cache_path: 'data/raw'
  secondary_timeframes:
    - '1h'
    - '30m'
  cache_expiration_days: 7
  use_cache: true
  refresh: false
  max_workers: 1
  # Optional time-aware test window used by CLI training
  test_start: '2024-09-01'
  test_end: '2024-12-31'
```

**Key Parameters:**
- `symbols`: List of stock symbols to process
- `start_date`/`end_date`: Data date range
- `cache_dir`: Directory for cached data
- `secondary_timeframes`: Optional list of higher-frequency bars to resample into the primary `timeframe` using OHLCV aggregations (`open→first`, `high→max`, `low→min`, `close→last`, `volume→sum`)
- `use_cache`: Enable/disable caching
- `refresh`: Force fresh data download
- `max_workers`: Parallel processing workers
- `test_start`/`test_end`: Optional test window for time-aware train/test split (if unset, last `training.test_size` fraction is used chronologically)

!!! info "Date validation and fallback"
    QuantTradeAI now validates that any configured `test_start`/`test_end` values fall within the overall `start_date` → `end_date` range and that the window is well ordered. If the requested window passes validation but your downloaded data is missing rows inside that range, the pipeline emits a warning and automatically falls back to the chronological `training.test_size` split so phase‑1 training can proceed.

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

Note: Hyperparameter tuning uses `TimeSeriesSplit(n_splits=cv_folds)` to avoid look‑ahead bias.
```

### Trading Parameters

```yaml
trading:
  initial_capital: 100000   # Starting capital for saved-model backtests
  position_size: 0.2
  stop_loss: 0.02
  take_profit: 0.04
  max_positions: 5
  transaction_cost: 0.001
  max_risk_per_trade: 0.02  # Optional overrides for portfolio manager sizing
  max_portfolio_risk: 0.10
```

### Backtest Execution

```yaml
execution:
  transaction_costs:
    enabled: true
    mode: bps         # bps or fixed
    value: 5          # 5 bps = 0.05%
    apply_on: notional
  slippage:
    enabled: true
    mode: bps
    value: 10
    reference_price: close  # or mid if available
  liquidity:
    enabled: false
    max_participation: 0.1
    volume_source: bar_volume
  impact:
    enabled: false
    model: linear        # linear, square_root, almgren_chriss
    alpha: 0.0
    beta: 0.0
    alpha_buy: 0.0       # optional asymmetric coefficients
    alpha_sell: 0.0
    decay: 0.0           # temporary impact decay rate
    decay_volume_coeff: 0.0
    spread: 0.0          # bid-ask spread per share
    spread_model: {type: dynamic}
    average_daily_volume: 0
  borrow_fee:
    enabled: false
    rate_bps: 0
  intrabar:
    enabled: false
    drift: 0.0
    volatility: 0.0
    synthetic_ticks: 0
```

The `impact` block activates market impact modeling. Parameters `alpha`/`beta`
and their buy/sell counterparts control the chosen model, while `decay`,
`decay_volume_coeff`, and `spread_model` enable dynamic spread and volume-based
decay. Default parameter sets per asset class can be defined in
`config/impact_config.yaml`.

The `borrow_fee` block applies financing costs to short positions, and the
`intrabar` block enables tick-level fill simulation with optional synthetic
Brownian motion ticks.

### Position Manager

```yaml
position_manager:
  risk_management:
    drawdown_protection:
      enabled: true
      max_drawdown_pct: 0.2
  impact:
    enabled: true
    model: linear
    alpha: 0.1
    beta: 0.05
  reconciliation:
    intraday: "1m"
    daily: "1d"
  mode: paper
```

Controls live position tracking and execution logic. The `impact` section
reuses backtest models, while `reconciliation` intervals harmonize intraday and
daily views. Set `mode` to `paper` or `live`.

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

## 🛡️ Risk Management

```yaml
risk_management:
  drawdown_protection:
    enabled: true
    max_drawdown_pct: 0.15
    warning_threshold: 0.8
    soft_stop_threshold: 0.9
    hard_stop_threshold: 1.0
  turnover_limits:
    daily_max: 2.0
    weekly_max: 5.0
    monthly_max: 15.0
```

**Key Parameters:**
- `drawdown_protection`: monitors portfolio equity and halts trading at specified levels
- `turnover_limits`: caps how frequently positions may change over each period

### CLI Usage

The backtesting CLI can enforce these limits via the drawdown guard:

```bash
poetry run quanttradeai backtest-model -m <model_dir> -c config/model_config.yaml --risk-config config/risk_config.yaml
```

If the path passed to `--risk-config` is missing or omitted, the command still runs, but no drawdown halts are applied.

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
