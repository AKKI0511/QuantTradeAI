# QuantTradeAI Quick Reference Guide

## Quick Start

### 1. Basic Setup

```bash
# Install dependencies
poetry install

# Fetch data for all symbols
poetry run quanttradeai fetch-data -c config/model_config.yaml

# Run complete training pipeline
poetry run quanttradeai train -c config/model_config.yaml
```

### 2. Python API Usage

```python
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.models.classifier import MomentumClassifier
from src.backtest.backtester import simulate_trades, compute_metrics

# Initialize components
loader = DataLoader("config/model_config.yaml")
processor = DataProcessor("config/features_config.yaml")
classifier = MomentumClassifier("config/model_config.yaml")

# Fetch and process data
data_dict = loader.fetch_data()
df = data_dict['AAPL']
df_processed = processor.process_data(df)
df_labeled = processor.generate_labels(df_processed)

# Train model
X, y = classifier.prepare_data(df_labeled)
classifier.train(X, y)

# Make predictions and backtest
predictions = classifier.predict(X)
df_labeled['predicted_label'] = predictions
df_trades = simulate_trades(df_labeled)
metrics = compute_metrics(df_trades)
```

## Common Patterns

### Data Loading

```python
# Load data for specific symbols
loader = DataLoader("config/model_config.yaml")
data = loader.fetch_data(symbols=['AAPL', 'TSLA'], refresh=True)

# Validate data
is_valid = loader.validate_data(data)

# Save data to custom location
loader.save_data(data, "data/custom_cache")
```

### Feature Engineering

```python
# Process raw OHLCV data
processor = DataProcessor("config/features_config.yaml")
df_processed = processor.process_data(raw_df)

# Generate labels with custom parameters
df_labeled = processor.generate_labels(df_processed, forward_returns=10, threshold=0.02)

# Check generated features
print(f"Total features: {len(df_processed.columns)}")
print(f"Feature names: {list(df_processed.columns)}")
```

### Model Training

```python
# Initialize and prepare data
classifier = MomentumClassifier("config/model_config.yaml")
X, y = classifier.prepare_data(df_labeled)

# Optimize hyperparameters
best_params = classifier.optimize_hyperparameters(X, y, n_trials=100)

# Train with optimized parameters
classifier.train(X, y, params=best_params)

# Evaluate model
metrics = classifier.evaluate(X, y)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Save model
classifier.save_model("models/trained/AAPL")
```

### Backtesting

```python
# Simulate trades with risk management
df_trades = simulate_trades(
    df_labeled, 
    stop_loss_pct=0.02, 
    take_profit_pct=0.04
)

# Calculate performance metrics
metrics = compute_metrics(df_trades, risk_free_rate=0.02)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
```

## Technical Indicators

```python
from src.features.technical import sma, ema, rsi, macd, stochastic

# Calculate moving averages
sma_20 = sma(df['Close'], 20)
ema_20 = ema(df['Close'], 20)

# Calculate momentum indicators
rsi_14 = rsi(df['Close'], 14)
macd_df = macd(df['Close'], fast=12, slow=26, signal=9)

# Calculate stochastic oscillator
stoch_df = stochastic(df['High'], df['Low'], df['Close'])
```

## Custom Features

```python
from src.features.custom import momentum_score, volatility_breakout

# Calculate momentum score
score = momentum_score(
    df['Close'], 
    df['sma_20'], 
    df['rsi'], 
    df['macd'], 
    df['macd_signal']
)

# Calculate volatility breakout
breakout = volatility_breakout(df['High'], df['Low'], df['Close'])
```

## Risk Management

```python
from src.trading.risk import apply_stop_loss_take_profit, position_size

# Apply stop-loss and take-profit
df_with_risk = apply_stop_loss_take_profit(
    df, 
    stop_loss_pct=0.02, 
    take_profit_pct=0.04
)

# Calculate position size
qty = position_size(
    capital=10000, 
    risk_per_trade=0.02, 
    stop_loss_pct=0.05, 
    price=150.0
)
```

## Performance Metrics

```python
from src.utils.metrics import classification_metrics, sharpe_ratio, max_drawdown

# Classification metrics
metrics = classification_metrics(y_true, y_pred)
print(f"F1 Score: {metrics['f1']:.4f}")

# Risk-adjusted returns
sharpe = sharpe_ratio(returns, risk_free_rate=0.02)
mdd = max_drawdown(equity_curve)
```

## Visualization

```python
from src.utils.visualization import plot_price, plot_performance

# Plot price chart
plot_price(df, title="AAPL Price Chart")

# Plot equity curve
plot_performance(equity_curve, title="Strategy Performance")
```

## Configuration Examples

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

## Error Handling

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

## Best Practices

### 1. Data Management
- Always validate data before processing
- Use caching to avoid repeated downloads
- Check for missing values and outliers

### 2. Model Training
- Use cross-validation for robust evaluation
- Optimize hyperparameters for each symbol
- Save models with descriptive names

### 3. Risk Management
- Always implement stop-loss and take-profit
- Calculate position sizes based on risk
- Monitor drawdown and Sharpe ratio

### 4. Performance Monitoring
- Track both classification and trading metrics
- Use multiple timeframes for evaluation
- Monitor out-of-sample performance

## Troubleshooting

### Common Issues

1. **Data Loading Errors**
   ```python
   # Check cache directory
   import os
   print(os.path.exists("data/raw"))
   
   # Force refresh
   data = loader.fetch_data(refresh=True)
   ```

2. **Feature Generation Errors**
   ```python
   # Check for NaN values
   print(df.isnull().sum())
   
   # Handle missing values
   df = df.fillna(method='ffill')
   ```

3. **Model Training Errors**
   ```python
   # Check data shapes
   print(f"X shape: {X.shape}")
   print(f"y shape: {y.shape}")
   
   # Check for class imbalance
   print(pd.Series(y).value_counts())
   ```

4. **Backtesting Errors**
   ```python
   # Check label distribution
   print(df['label'].value_counts())
   
   # Ensure proper date index
   print(df.index.dtype)
   ```

This quick reference guide provides the most common usage patterns and examples for the QuantTradeAI framework.