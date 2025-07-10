# QuantTradeAI API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Command Line Interface](#command-line-interface)
3. [Data Loading and Processing](#data-loading-and-processing)
4. [Feature Engineering](#feature-engineering)
5. [Machine Learning Models](#machine-learning-models)
6. [Backtesting Framework](#backtesting-framework)
7. [Trading Utilities](#trading-utilities)
8. [Utility Functions](#utility-functions)
9. [Configuration](#configuration)

## Overview

QuantTradeAI is a comprehensive machine learning framework for quantitative trading strategies, specifically designed for momentum trading using ensemble models. The framework provides a complete pipeline from data acquisition to model training and backtesting.

## Command Line Interface

### Main Entry Point

The primary CLI interface is accessible through the `main.py` module.

#### `run_pipeline(config_path: str = "config/model_config.yaml")`

Runs the complete trading strategy pipeline including data fetching, feature engineering, model training, and evaluation.

**Parameters:**
- `config_path` (str): Path to the model configuration file

**Returns:**
- `dict`: Results dictionary containing hyperparameters and metrics for each symbol

**Example:**
```python
from src.main import run_pipeline

# Run the complete pipeline
results = run_pipeline("config/model_config.yaml")
print(f"Training completed for {len(results)} symbols")
```

#### `fetch_data_only(config_path: str, refresh: bool = False)`

Fetches and caches OHLCV data for all configured symbols.

**Parameters:**
- `config_path` (str): Path to the configuration file
- `refresh` (bool): Force refresh of cached data

**Example:**
```python
from src.main import fetch_data_only

# Fetch data for all symbols
fetch_data_only("config/model_config.yaml", refresh=True)
```

#### `evaluate_model(config_path: str, model_path: str)`

Loads a saved model and evaluates it on the configured dataset.

**Parameters:**
- `config_path` (str): Path to the configuration file
- `model_path` (str): Directory containing the saved model

**Example:**
```python
from src.main import evaluate_model

# Evaluate a trained model
evaluate_model("config/model_config.yaml", "models/trained/AAPL")
```

### CLI Commands

The framework provides several command-line interfaces:

```bash
# Show help
poetry run quanttradeai --help

# Fetch data and cache it
poetry run quanttradeai fetch-data -c config/model_config.yaml

# Run full training pipeline
poetry run quanttradeai train -c config/model_config.yaml

# Evaluate a saved model
poetry run quanttradeai evaluate -c config/model_config.yaml -m models/trained/AAPL
```

## Data Loading and Processing

### DataLoader Class

#### `DataLoader(config_path: str = "config/model_config.yaml", data_source: Optional[DataSource] = None)`

Handles data fetching, caching, and validation for multiple financial instruments.

**Parameters:**
- `config_path` (str): Path to configuration file
- `data_source` (DataSource, optional): Custom data source implementation

**Example:**
```python
from src.data.loader import DataLoader

# Initialize with default configuration
loader = DataLoader("config/model_config.yaml")

# Fetch data for all symbols
data_dict = loader.fetch_data()

# Fetch data for specific symbols
data_dict = loader.fetch_data(symbols=['AAPL', 'META'], refresh=True)
```

#### `fetch_data(symbols: Optional[List[str]] = None, refresh: Optional[bool] = None) -> Dict[str, pd.DataFrame]`

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

#### `validate_data(data_dict: Dict[str, pd.DataFrame]) -> bool`

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

#### `save_data(data_dict: Dict[str, pd.DataFrame], path: Optional[str] = None)`

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

### DataSource Classes

#### `DataSource` (Abstract Base Class)

Abstract interface for price data providers.

#### `YFinanceDataSource`

DataSource implementation using the yfinance package.

**Example:**
```python
from src.data.datasource import YFinanceDataSource

# Initialize YFinance data source
data_source = YFinanceDataSource()

# Fetch data for a symbol
df = data_source.fetch("AAPL", "2023-01-01", "2023-12-31")
```

#### `AlphaVantageDataSource(api_key: Optional[str] = None)`

DataSource implementation for AlphaVantage API.

**Parameters:**
- `api_key` (str, optional): AlphaVantage API key. If None, reads from environment variable

**Example:**
```python
from src.data.datasource import AlphaVantageDataSource

# Initialize with API key
data_source = AlphaVantageDataSource("YOUR_API_KEY")

# Fetch data
df = data_source.fetch("AAPL", "2023-01-01", "2023-12-31")
```

### DataProcessor Class

#### `DataProcessor(config_path: str = "config/features_config.yaml")`

Processes raw OHLCV data and generates required features for the trading strategy.

**Parameters:**
- `config_path` (str): Path to feature configuration file

**Example:**
```python
from src.data.processor import DataProcessor

# Initialize processor
processor = DataProcessor("config/features_config.yaml")

# Process raw data
processed_df = processor.process_data(raw_df)
```

#### `process_data(data: pd.DataFrame) -> pd.DataFrame`

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

#### `generate_labels(df: pd.DataFrame, forward_returns: int = 5, threshold: float = 0.01) -> pd.DataFrame`

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

## Feature Engineering

### Technical Indicators (`src.features.technical`)

#### `sma(series: pd.Series, period: int) -> pd.Series`

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

#### `ema(series: pd.Series, period: int) -> pd.Series`

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

#### `rsi(series: pd.Series, period: int = 14) -> pd.Series`

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

#### `macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame`

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

#### `stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame`

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

### Custom Features (`src.features.custom`)

#### `momentum_score(close: pd.Series, sma: pd.Series, rsi_series: pd.Series, macd: pd.Series, macd_signal: pd.Series) -> pd.Series`

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

#### `volatility_breakout(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 20, threshold: float = 2.0) -> pd.Series`

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

## Machine Learning Models

### MomentumClassifier Class

#### `MomentumClassifier(config_path: str = "config/model_config.yaml")`

Voting Classifier for momentum trading strategy using Logistic Regression, Random Forest, and XGBoost.

**Parameters:**
- `config_path` (str): Path to model configuration file

**Example:**
```python
from src.models.classifier import MomentumClassifier

# Initialize classifier
classifier = MomentumClassifier("config/model_config.yaml")
```

#### `prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]`

Prepares data for training/prediction.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with features and labels

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Tuple of features array and labels array

**Example:**
```python
# Prepare data for training
X, y = classifier.prepare_data(labeled_df)
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
```

#### `optimize_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 100) -> Dict`

Optimizes hyperparameters using Optuna.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Labels array
- `n_trials` (int): Number of optimization trials

**Returns:**
- `Dict`: Dictionary of best parameters

**Example:**
```python
# Optimize hyperparameters
best_params = classifier.optimize_hyperparameters(X_train, y_train, n_trials=50)
print(f"Best parameters: {best_params}")
```

#### `train(X: np.ndarray, y: np.ndarray, params: Dict[str, Any] = None)`

Trains the voting classifier.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): Labels array
- `params` (Dict[str, Any], optional): Hyperparameters

**Example:**
```python
# Train with optimized parameters
classifier.train(X_train, y_train, params=best_params)

# Train with default parameters
classifier.train(X_train, y_train)
```

#### `predict(X: np.ndarray) -> np.ndarray`

Makes predictions using the trained model.

**Parameters:**
- `X` (np.ndarray): Feature matrix

**Returns:**
- `np.ndarray`: Array of predictions

**Example:**
```python
# Make predictions
predictions = classifier.predict(X_test)
print(f"Predictions: {predictions}")
```

#### `evaluate(X: np.ndarray, y: np.ndarray) -> Dict[str, float]`

Evaluates model performance.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): True labels

**Returns:**
- `Dict[str, float]`: Dictionary of performance metrics

**Example:**
```python
# Evaluate model
metrics = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

#### `save_model(path: str)`

Saves the trained model and scaler.

**Parameters:**
- `path` (str): Directory path to save model

**Example:**
```python
# Save model
classifier.save_model("models/trained/AAPL")
```

#### `load_model(path: str)`

Loads a trained model and scaler.

**Parameters:**
- `path` (str): Directory path containing saved model

**Example:**
```python
# Load model
classifier.load_model("models/trained/AAPL")
```

## Backtesting Framework

### `simulate_trades(df: pd.DataFrame, stop_loss_pct: float | None = None, take_profit_pct: float | None = None) -> pd.DataFrame`

Simulates trades using label signals.

**Parameters:**
- `df` (pd.DataFrame): DataFrame containing Close prices and label column
- `stop_loss_pct` (float, optional): Stop-loss percentage as decimal
- `take_profit_pct` (float, optional): Take-profit percentage as decimal

**Returns:**
- `pd.DataFrame`: DataFrame with additional strategy_return and equity_curve columns

**Example:**
```python
from src.backtest.backtester import simulate_trades

# Simulate trades with stop-loss and take-profit
df_with_trades = simulate_trades(df, stop_loss_pct=0.02, take_profit_pct=0.04)

# Simulate trades without risk management
df_with_trades = simulate_trades(df)
```

### `compute_metrics(data: pd.DataFrame, risk_free_rate: float = 0.0) -> dict`

Calculates basic performance metrics for a strategy.

**Parameters:**
- `data` (pd.DataFrame): Output from simulate_trades
- `risk_free_rate` (float): Annual risk-free rate for Sharpe ratio

**Returns:**
- `dict`: Dictionary with cumulative_return, sharpe_ratio, and max_drawdown

**Example:**
```python
from src.backtest.backtester import compute_metrics

# Calculate performance metrics
metrics = compute_metrics(df_with_trades, risk_free_rate=0.02)
print(f"Cumulative Return: {metrics['cumulative_return']:.4f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
```

## Trading Utilities

### Risk Management

#### `apply_stop_loss_take_profit(df: pd.DataFrame, stop_loss_pct: float | None = None, take_profit_pct: float | None = None) -> pd.DataFrame`

Applies stop-loss and take-profit rules to trading signals.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with Close prices and label column
- `stop_loss_pct` (float, optional): Stop-loss percentage as decimal
- `take_profit_pct` (float, optional): Take-profit percentage as decimal

**Returns:**
- `pd.DataFrame`: DataFrame with adjusted label column

**Example:**
```python
from src.trading.risk import apply_stop_loss_take_profit

# Apply 2% stop-loss and 4% take-profit
df_with_risk = apply_stop_loss_take_profit(df, stop_loss_pct=0.02, take_profit_pct=0.04)
```

#### `position_size(capital: float, risk_per_trade: float, stop_loss_pct: float, price: float) -> int`

Calculates position size based on account risk parameters.

**Parameters:**
- `capital` (float): Available capital
- `risk_per_trade` (float): Risk per trade as decimal
- `stop_loss_pct` (float): Stop-loss percentage as decimal
- `price` (float): Current price

**Returns:**
- `int`: Position size in units

**Example:**
```python
from src.trading.risk import position_size

# Calculate position size
qty = position_size(capital=10000, risk_per_trade=0.02, stop_loss_pct=0.05, price=150.0)
print(f"Position size: {qty} shares")
```

## Utility Functions

### Performance Metrics

#### `classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict`

Returns basic classification metrics.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels

**Returns:**
- `dict`: Dictionary with accuracy, precision, recall, and f1 score

**Example:**
```python
from src.utils.metrics import classification_metrics

# Calculate classification metrics
metrics = classification_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

#### `sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float`

Calculates annualized Sharpe ratio.

**Parameters:**
- `returns` (pd.Series): Return series
- `risk_free_rate` (float): Annual risk-free rate

**Returns:**
- `float`: Annualized Sharpe ratio

**Example:**
```python
from src.utils.metrics import sharpe_ratio

# Calculate Sharpe ratio
sharpe = sharpe_ratio(returns, risk_free_rate=0.02)
print(f"Sharpe Ratio: {sharpe:.4f}")
```

#### `max_drawdown(equity_curve: pd.Series) -> float`

Computes maximum drawdown from an equity curve.

**Parameters:**
- `equity_curve` (pd.Series): Cumulative equity curve

**Returns:**
- `float`: Maximum drawdown as decimal

**Example:**
```python
from src.utils.metrics import max_drawdown

# Calculate maximum drawdown
mdd = max_drawdown(equity_curve)
print(f"Maximum Drawdown: {mdd:.4f}")
```

### Visualization

#### `plot_price(data: pd.DataFrame, title: str = "Price Chart")`

Plots OHLC closing price.

**Parameters:**
- `data` (pd.DataFrame): DataFrame with OHLC data
- `title` (str): Plot title

**Example:**
```python
from src.utils.visualization import plot_price

# Plot price chart
plot_price(df, title="AAPL Price Chart")
```

#### `plot_performance(equity_curve: pd.Series, title: str = "Equity Curve")`

Plots cumulative returns or equity curve.

**Parameters:**
- `equity_curve` (pd.Series): Equity curve series
- `title` (str): Plot title

**Example:**
```python
from src.utils.visualization import plot_performance

# Plot equity curve
plot_performance(equity_curve, title="Strategy Performance")
```

## Configuration

### Model Configuration (`config/model_config.yaml`)

The model configuration file contains settings for:

- **Data Parameters**: Symbols, date ranges, caching settings
- **Feature Engineering**: Technical indicator parameters
- **Model Parameters**: Hyperparameters for each model type
- **Training Parameters**: Test size, cross-validation settings
- **Trading Parameters**: Position sizing, risk management

**Example Configuration:**
```yaml
data:
  symbols: ['AAPL', 'META', 'TSLA', 'JPM', 'AMZN']
  start_date: '2015-01-01'
  end_date: '2024-12-31'
  cache_dir: 'data/raw'
  use_cache: true
  refresh: false

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
    class_weight: 'balanced'
  
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
```

### Feature Configuration (`config/features_config.yaml`)

The feature configuration file contains settings for:

- **Price Features**: Technical indicators and their parameters
- **Volume Features**: Volume-based indicators
- **Volatility Features**: Bollinger Bands, ATR settings
- **Custom Features**: Strategy-specific features
- **Feature Selection**: Selection methods and parameters
- **Preprocessing**: Scaling and outlier handling

**Example Configuration:**
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
    target_range: [-1, 1]
  
  outliers:
    method: 'winsorize'
    limits: [0.01, 0.99]
```

## Complete Example Workflow

Here's a complete example of how to use the QuantTradeAI framework:

```python
import pandas as pd
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.models.classifier import MomentumClassifier
from src.backtest.backtester import simulate_trades, compute_metrics
from sklearn.model_selection import train_test_split

# 1. Initialize components
loader = DataLoader("config/model_config.yaml")
processor = DataProcessor("config/features_config.yaml")
classifier = MomentumClassifier("config/model_config.yaml")

# 2. Fetch and process data
data_dict = loader.fetch_data()
results = {}

for symbol, df in data_dict.items():
    print(f"Processing {symbol}...")
    
    # 3. Generate features
    df_processed = processor.process_data(df)
    
    # 4. Generate labels
    df_labeled = processor.generate_labels(df_processed)
    
    # 5. Prepare data for training
    X, y = classifier.prepare_data(df_labeled)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 6. Optimize and train model
    best_params = classifier.optimize_hyperparameters(X_train, y_train, n_trials=50)
    classifier.train(X_train, y_train, params=best_params)
    
    # 7. Evaluate model
    train_metrics = classifier.evaluate(X_train, y_train)
    test_metrics = classifier.evaluate(X_test, y_test)
    
    # 8. Generate predictions and backtest
    predictions = classifier.predict(X_test)
    df_test = df_labeled.iloc[-len(X_test):].copy()
    df_test['predicted_label'] = predictions
    
    # 9. Simulate trades
    df_trades = simulate_trades(df_test, stop_loss_pct=0.02, take_profit_pct=0.04)
    
    # 10. Calculate performance metrics
    performance = compute_metrics(df_trades)
    
    results[symbol] = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'performance': performance
    }
    
    print(f"{symbol} Results:")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {performance['max_drawdown']:.4f}")

# Save results
import json
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)
```

This documentation provides comprehensive coverage of all public APIs, functions, and components in the QuantTradeAI framework, including examples and usage instructions for each major component.