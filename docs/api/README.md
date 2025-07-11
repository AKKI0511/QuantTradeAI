# API Reference

Complete API documentation for QuantTradeAI.

## 📚 API Sections

### Core Components
- **[Data Loading](data.md)** - DataLoader, DataProcessor, DataSource classes
- **[Feature Engineering](features.md)** - Technical indicators and custom features
- **[Machine Learning](models.md)** - MomentumClassifier and training utilities
- **[Backtesting](backtesting.md)** - Trade simulation and performance metrics
- **[Trading Utilities](trading.md)** - Risk management and position sizing

## 🚀 Quick Navigation

### By Task
- **Data Management**: [Data Loading](data.md)
- **Feature Creation**: [Feature Engineering](features.md)
- **Model Training**: [Machine Learning](models.md)
- **Strategy Testing**: [Backtesting](backtesting.md)
- **Risk Management**: [Trading Utilities](trading.md)

### By Component
- **DataLoader**: [Data Loading](data.md#dataloader-class)
- **DataProcessor**: [Data Loading](data.md#dataprocessor-class)
- **MomentumClassifier**: [Machine Learning](models.md#momentumclassifier-class)
- **Technical Indicators**: [Feature Engineering](features.md#technical-indicators)
- **Backtesting Functions**: [Backtesting](backtesting.md)
- **Risk Management**: [Trading Utilities](trading.md)

## 📖 Usage Examples

### Basic Workflow
```python
from quanttradeai.data.loader import DataLoader
from quanttradeai.data.processor import DataProcessor
from quanttradeai.models.classifier import MomentumClassifier

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
```

### Advanced Usage
```python
from quanttradeai.backtest.backtester import simulate_trades, compute_metrics
from quanttradeai.trading.risk import apply_stop_loss_take_profit

# Backtest with risk management
df_with_risk = apply_stop_loss_take_profit(df_labeled, stop_loss_pct=0.02)
df_trades = simulate_trades(df_with_risk)
metrics = compute_metrics(df_trades)
```

## 🔍 Finding Functions

### Data Functions
- `DataLoader.fetch_data()` - Fetch OHLCV data
- `DataLoader.validate_data()` - Validate data quality
- `DataProcessor.process_data()` - Generate features
- `DataProcessor.generate_labels()` - Create trading signals

### Feature Functions
- `sma()`, `ema()` - Moving averages
- `rsi()`, `macd()` - Momentum indicators
- `momentum_score()` - Custom momentum score
- `volatility_breakout()` - Volatility signals

### Model Functions
- `MomentumClassifier.train()` - Train ensemble model
- `MomentumClassifier.predict()` - Make predictions
- `MomentumClassifier.evaluate()` - Evaluate performance
- `MomentumClassifier.optimize_hyperparameters()` - Tune parameters

### Backtesting Functions
- `simulate_trades()` - Simulate trading
- `compute_metrics()` - Calculate performance metrics

### Utility Functions
- `classification_metrics()` - Classification performance
- `sharpe_ratio()`, `max_drawdown()` - Trading metrics
- `plot_price()`, `plot_performance()` - Visualization

## 📚 Related Documentation

- **[Getting Started](../getting-started.md)** - Installation and first steps
- **[Quick Reference](../quick-reference.md)** - Common patterns and commands
- **[Configuration](../configuration.md)** - Configuration guide
- **[Examples](../examples/)** - Usage examples and tutorials