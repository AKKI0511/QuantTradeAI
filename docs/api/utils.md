# Utility Modules
This section documents utility modules in quanttradeai/utils/ which were previously under-documented.

# Utilities API
Utility functions and classes for QuantTradeAI configuration, metrics, and visualization.

## 📋 Overview
The `quanttradeai.utils` module provides essential utility functions for:
- **Configuration Management**: Pydantic schema validation for configurations
- **Performance Metrics**: Classification and trading performance analysis
- **Data Visualization**: Plotting functions for price data and performance analysis

## Configuration Schemas (config_schemas.py)

Defines Pydantic data models for type-safe config files. Used for validating data, model, pipeline, and feature configuration structures.

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

## Performance Metrics (metrics.py)
This module provides evaluation metrics for classification and trading strategies.

- `classification_metrics(y_true, y_pred)`: Basic classification metrics (accuracy, precision, recall, F1).
- `sharpe_ratio(returns, risk_free_rate=0.0)`: Annualized Sharpe ratio.
- `max_drawdown(equity_curve)`: Compute maximum drawdown.

## Data Visualization (visualization.py)
This module provides simple utilities for displaying price and performance: 

- `plot_price(data, title="Price Chart")`: Line plot of closing prices.
- `plot_performance(equity_curve, title="Equity Curve")`: Line plot of portfolio equity or cumulative returns.
