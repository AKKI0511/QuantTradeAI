# Utilities API

Utility functions and classes for QuantTradeAI configuration, metrics, and visualization.

## 📋 Overview

The `quanttradeai.utils` module provides essential utility functions for:
- **Configuration Management**: Pydantic schema validation for configurations
- **Performance Metrics**: Classification and trading performance analysis
- **Data Visualization**: Plotting functions for price data and performance analysis

---

## 🔧 Configuration Schemas (`config_schemas.py`)

Pydantic schema classes for configuration validation and type checking.

### DataSection

**Class**: `DataSection`

Validates data configuration parameters.

**Schema Fields**:
```python
# Example data section configuration
data_config = {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "period": "2y",
    "interval": "1d",
    "source": "yahoo"
}
```

**Usage**:
```python
from quanttradeai.utils.config_schemas import DataSection

# Validate data configuration
data_section = DataSection(**data_config)
print(data_section.symbols)  # ['AAPL', 'MSFT', 'GOOGL']
```

### ModelConfigSchema

**Class**: `ModelConfigSchema`

Validates machine learning model configuration parameters.

**Schema Fields**:
```python
# Example model configuration
model_config = {
    "algorithms": ["random_forest", "gradient_boosting"],
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42
}
```

**Usage**:
```python
from quanttradeai.utils.config_schemas import ModelConfigSchema

# Validate model configuration
model_schema = ModelConfigSchema(**model_config)
print(model_schema.algorithms)  # ['random_forest', 'gradient_boosting']
```

### PipelineConfig

**Class**: `PipelineConfig`

Validates end-to-end pipeline configuration.

**Schema Fields**:
```python
# Example pipeline configuration
pipeline_config = {
    "data": data_config,
    "features": features_config,
    "model": model_config,
    "backtest": backtest_config
}
```

**Usage**:
```python
from quanttradeai.utils.config_schemas import PipelineConfig

# Validate complete pipeline configuration
pipeline = PipelineConfig(**pipeline_config)
print(pipeline.data.symbols)  # Access nested configuration
```

### FeaturesConfigSchema

**Class**: `FeaturesConfigSchema`

Validates feature engineering configuration parameters.

**Schema Fields**:
```python
# Example features configuration
features_config = {
    "technical_indicators": ["sma", "ema", "rsi", "macd"],
    "lookback_periods": [5, 10, 20, 50],
    "custom_features": ["momentum_score", "volatility_breakout"]
}
```

**Usage**:
```python
from quanttradeai.utils.config_schemas import FeaturesConfigSchema

# Validate features configuration
features_schema = FeaturesConfigSchema(**features_config)
print(features_schema.technical_indicators)  # ['sma', 'ema', 'rsi', 'macd']
```

---

## 📊 Performance Metrics (`metrics.py`)

Functions for calculating classification and trading performance metrics.

### Classification Metrics

#### accuracy()

**Function**: `accuracy(y_true, y_pred)`

Calculate classification accuracy.

**Parameters**:
- `y_true` (array-like): True labels
- `y_pred` (array-like): Predicted labels

**Returns**:
- `float`: Accuracy score (0.0 to 1.0)

**Example**:
```python
from quanttradeai.utils.metrics import accuracy

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 0]
acc = accuracy(y_true, y_pred)
print(f"Accuracy: {acc:.3f}")  # Accuracy: 0.800
```

#### precision()

**Function**: `precision(y_true, y_pred, average='binary')`

Calculate precision score.

**Parameters**:
- `y_true` (array-like): True labels
- `y_pred` (array-like): Predicted labels
- `average` (str): Averaging strategy ('binary', 'macro', 'micro', 'weighted')

**Returns**:
- `float`: Precision score

**Example**:
```python
from quanttradeai.utils.metrics import precision

precision_score = precision(y_true, y_pred)
print(f"Precision: {precision_score:.3f}")
```

#### recall()

**Function**: `recall(y_true, y_pred, average='binary')`

Calculate recall score.

**Parameters**:
- `y_true` (array-like): True labels
- `y_pred` (array-like): Predicted labels
- `average` (str): Averaging strategy

**Returns**:
- `float`: Recall score

**Example**:
```python
from quanttradeai.utils.metrics import recall

recall_score = recall(y_true, y_pred)
print(f"Recall: {recall_score:.3f}")
```

#### f1_score()

**Function**: `f1_score(y_true, y_pred, average='binary')`

Calculate F1 score (harmonic mean of precision and recall).

**Parameters**:
- `y_true` (array-like): True labels
- `y_pred` (array-like): Predicted labels
- `average` (str): Averaging strategy

**Returns**:
- `float`: F1 score

**Example**:
```python
from quanttradeai.utils.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.3f}")
```

### Trading Performance Metrics

#### sharpe_ratio()

**Function**: `sharpe_ratio(returns, risk_free_rate=0.02)`

Calculate Sharpe ratio for risk-adjusted returns.

**Parameters**:
- `returns` (array-like): Portfolio returns
- `risk_free_rate` (float): Risk-free rate (default: 0.02)

**Returns**:
- `float`: Sharpe ratio

**Example**:
```python
from quanttradeai.utils.metrics import sharpe_ratio
import numpy as np

# Sample daily returns
returns = np.array([0.01, -0.005, 0.02, 0.001, -0.01])
sharpe = sharpe_ratio(returns)
print(f"Sharpe Ratio: {sharpe:.3f}")
```

#### max_drawdown()

**Function**: `max_drawdown(equity_curve)`

Calculate maximum drawdown from equity curve.

**Parameters**:
- `equity_curve` (array-like): Cumulative portfolio value over time

**Returns**:
- `float`: Maximum drawdown (negative value)

**Example**:
```python
from quanttradeai.utils.metrics import max_drawdown
import numpy as np

# Sample equity curve
equity = np.array([1000, 1100, 1050, 900, 950, 1200])
mdd = max_drawdown(equity)
print(f"Max Drawdown: {mdd:.3f}")
```

---

## 📈 Visualization (`visualization.py`)

Functions for plotting price data and performance analysis.

### plot_price()

**Function**: `plot_price(df, title="Price Chart", figsize=(12, 6))`

Plot OHLCV price data with technical indicators.

**Parameters**:
- `df` (pandas.DataFrame): DataFrame with OHLCV data
- `title` (str): Chart title (default: "Price Chart")
- `figsize` (tuple): Figure size (default: (12, 6))

**Returns**:
- `matplotlib.figure.Figure`: Figure object

**Example**:
```python
from quanttradeai.utils.visualization import plot_price
import pandas as pd

# Assuming df has OHLCV columns
fig = plot_price(df, title="AAPL Price Analysis")
fig.show()
```

**Features**:
- Candlestick or line plots for price data
- Volume subplot
- Technical indicator overlays
- Customizable styling and colors

### plot_performance()

**Function**: `plot_performance(df_trades, benchmark=None, figsize=(15, 10))`

Plot comprehensive performance analysis including equity curve, drawdown, and metrics.

**Parameters**:
- `df_trades` (pandas.DataFrame): DataFrame with trade results
- `benchmark` (pandas.Series, optional): Benchmark returns for comparison
- `figsize` (tuple): Figure size (default: (15, 10))

**Returns**:
- `matplotlib.figure.Figure`: Figure object with multiple subplots

**Example**:
```python
from quanttradeai.utils.visualization import plot_performance

# Plot strategy performance
fig = plot_performance(df_trades, benchmark=sp500_returns)
fig.show()
```

**Subplots Include**:
1. **Equity Curve**: Cumulative portfolio value over time
2. **Drawdown Chart**: Underwater equity curve showing drawdowns
3. **Returns Distribution**: Histogram of daily/monthly returns
4. **Rolling Metrics**: Rolling Sharpe ratio and volatility
5. **Monthly Returns Heatmap**: Calendar view of monthly performance

**Performance Metrics Displayed**:
- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Average Win/Loss
- Profit Factor

---

## 🚀 Quick Examples

### Complete Classification Evaluation

```python
from quanttradeai.utils.metrics import accuracy, precision, recall, f1_score

# Evaluate model predictions
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

print(f"Accuracy: {accuracy(y_true, y_pred):.3f}")
print(f"Precision: {precision(y_true, y_pred):.3f}")
print(f"Recall: {recall(y_true, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.3f}")
```

### Trading Performance Analysis

```python
from quanttradeai.utils.metrics import sharpe_ratio, max_drawdown
from quanttradeai.utils.visualization import plot_performance
import numpy as np

# Calculate trading metrics
returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
equity_curve = np.cumprod(1 + returns) * 10000  # Starting with $10,000

sharpe = sharpe_ratio(returns)
mdd = max_drawdown(equity_curve)

print(f"Sharpe Ratio: {sharpe:.3f}")
print(f"Max Drawdown: {mdd:.3%}")

# Create performance visualization
df_trades = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=252),
    'equity': equity_curve,
    'returns': returns
})

fig = plot_performance(df_trades)
fig.show()
```

### Configuration Validation

```python
from quanttradeai.utils.config_schemas import (
    DataSection, ModelConfigSchema, FeaturesConfigSchema, PipelineConfig
)

# Validate complete configuration
config = {
    "data": {
        "symbols": ["AAPL", "MSFT"],
        "period": "1y",
        "interval": "1d"
    },
    "features": {
        "technical_indicators": ["sma", "rsi"],
        "lookback_periods": [10, 20]
    },
    "model": {
        "algorithms": ["random_forest"],
        "test_size": 0.2,
        "cv_folds": 5
    }
}

# This will validate all nested configurations
pipeline_config = PipelineConfig(**config)
print("Configuration validated successfully!")
```

---

## 🔗 Integration with Other Modules

The utilities module integrates seamlessly with other QuantTradeAI components:

- **Data Processing**: Use configuration schemas to validate data loading parameters
- **Feature Engineering**: Validate feature configuration before processing
- **Model Training**: Evaluate model performance with classification metrics
- **Backtesting**: Analyze trading performance with financial metrics and visualization
- **Risk Management**: Visualize drawdowns and risk metrics

---

## 📚 See Also

- [Data Loading](data.md) - Data management and processing
- [Feature Engineering](features.md) - Technical indicators and features
- [Machine Learning](models.md) - Model training and evaluation
- [Backtesting](backtesting.md) - Strategy testing and simulation
- [Trading Utilities](trading.md) - Risk management and position sizing
