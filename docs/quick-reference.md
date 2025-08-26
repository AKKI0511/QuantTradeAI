# Quick Reference

Common commands, patterns, and examples for QuantTradeAI.

## 🚀 CLI Commands

```bash
# Show help
poetry run quanttradeai --help

# Fetch data
poetry run quanttradeai fetch-data
poetry run quanttradeai fetch-data --refresh

# Train models
poetry run quanttradeai train

# Evaluate model
poetry run quanttradeai evaluate -m models/trained/AAPL

# Run backtest
poetry run quanttradeai backtest --config config/backtest_config.yaml
poetry run quanttradeai backtest --cost-bps 5 --slippage-bps 10
```

## 📊 Python API Patterns

### Data Loading
```python
from quanttradeai import DataLoader

# Initialize and fetch data
loader = DataLoader("config/model_config.yaml")
data = loader.fetch_data(symbols=['AAPL', 'TSLA'], refresh=True)

# Validate data
is_valid = loader.validate_data(data)
```

### Feature Engineering
```python
from quanttradeai import DataProcessor

# Process raw data
processor = DataProcessor("config/features_config.yaml")
df_processed = processor.process_data(raw_df)

# Generate labels
df_labeled = processor.generate_labels(df_processed, forward_returns=5, threshold=0.01)
```

### Model Training
```python
from quanttradeai import MomentumClassifier

# Initialize and train
classifier = MomentumClassifier("config/model_config.yaml")
X, y = classifier.prepare_data(df_labeled)
classifier.train(X, y)

# Save model
classifier.save_model("models/trained/AAPL")
```

### Backtesting
```python
from quanttradeai import simulate_trades, compute_metrics

# Simulate trades
df_trades = simulate_trades(df_labeled, stop_loss_pct=0.02, take_profit_pct=0.04)

# Calculate metrics
metrics = compute_metrics(df_trades, risk_free_rate=0.02)
```

## 🔧 Technical Indicators

```python
from quanttradeai.features import technical as ta

# Moving averages
sma_20 = ta.sma(df['Close'], 20)
ema_20 = ta.ema(df['Close'], 20)

# Momentum indicators
rsi_14 = ta.rsi(df['Close'], 14)
macd_df = ta.macd(df['Close'])
stoch_df = ta.stochastic(df['High'], df['Low'], df['Close'])
```

## 🛡️ Risk Management

```python
from quanttradeai import apply_stop_loss_take_profit, position_size

# Apply risk rules
df_with_risk = apply_stop_loss_take_profit(df, stop_loss_pct=0.02, take_profit_pct=0.04)

# Calculate position size
qty = position_size(capital=10000, risk_per_trade=0.02, stop_loss_pct=0.05, price=150.0)
```

```python
from quanttradeai import PortfolioManager

# Allocate capital across multiple symbols
pm = PortfolioManager(10000)
pm.open_position('AAPL', price=150, stop_loss_pct=0.05)
pm.open_position('TSLA', price=250, stop_loss_pct=0.05)
print(f"Portfolio exposure: {pm.risk_exposure:.2%}")
```

## 📈 Performance Metrics

```python
from quanttradeai.utils.metrics import classification_metrics, sharpe_ratio, max_drawdown

# Classification metrics
metrics = classification_metrics(y_true, y_pred)

# Trading metrics
sharpe = sharpe_ratio(returns, risk_free_rate=0.02)
mdd = max_drawdown(equity_curve)
```

## 📊 Visualization

```python
from quanttradeai.utils.visualization import plot_price, plot_performance

# Plot charts
plot_price(df, title="AAPL Price Chart")
plot_performance(equity_curve, title="Strategy Performance")
```

## ⚙️ Configuration Examples

### Model Configuration
```yaml
data:
  symbols: ['AAPL', 'META', 'TSLA']
  start_date: '2020-01-01'
  end_date: '2024-12-31'
  cache_dir: 'data/raw'
  use_cache: true

models:
  voting_classifier:
    voting: 'soft'
  
  logistic_regression:
    C: 1.0
    class_weight: 'balanced'
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    class_weight: 'balanced'
  
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
```

### Feature Configuration
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

volatility_features:
  bollinger_bands:
    period: 20
    std_dev: 2

preprocessing:
  scaling:
    method: 'standard'
  outliers:
    method: 'winsorize'
    limits: [0.01, 0.99]
```

## 🔌 Streaming

```python
from quanttradeai.streaming import StreamingGateway

gw = StreamingGateway("config/streaming.yaml")
gw.subscribe_to_trades(["AAPL"], callback=lambda m: print(m))
# gw.start_streaming()  # blocking
```

`config/streaming.yaml` example:

```yaml
streaming:
  symbols: ["AAPL"]
  providers:
    - name: "alpaca"
      websocket_url: "wss://stream.data.alpaca.markets/v2/iex"
      auth_method: "api_key"
      subscriptions: ["trades", "quotes"]
  buffer_size: 1000
  reconnect_attempts: 3
  health_check_interval: 30
```

## 🚨 Error Handling

```python
try:
    # Fetch data
    data = loader.fetch_data()
    
    # Process data
    df_processed = processor.process_data(data['AAPL'])
    
    # Train model
    classifier.train(X, y)
    
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🔍 Troubleshooting

### Data Issues
```python
# Check cache directory
import os
print(os.path.exists("data/raw"))

# Force refresh
data = loader.fetch_data(refresh=True)

# Check for NaN values
print(df.isnull().sum())
df = df.fillna(method='ffill')
```

### Model Issues
```python
# Check data shapes
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Check class distribution
print(pd.Series(y).value_counts())
```

### Backtesting Issues
```python
# Check label distribution
print(df['label'].value_counts())

# Ensure proper date index
print(df.index.dtype)
```

## 📚 Related Documentation

- **[Getting Started](getting-started.md)** - Installation and first steps
- **[API Reference](api/)** - Complete API documentation

- **[Configuration](configuration.md)** - Configuration guide
