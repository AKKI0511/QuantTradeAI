# Backtesting Framework

API documentation for trade simulation and performance metrics.

## Trade Simulation

### `simulate_trades(df: pd.DataFrame | dict[str, pd.DataFrame], stop_loss_pct: float | None = None, take_profit_pct: float | None = None, transaction_cost: float = 0.0, slippage: float = 0.0, portfolio: PortfolioManager | None = None) -> pd.DataFrame | dict[str, pd.DataFrame]`

Simulates trades using label signals.

**Parameters:**
- `df` (pd.DataFrame or dict[str, pd.DataFrame]): Single DataFrame or mapping of symbol to DataFrame with Close prices and label column
- `stop_loss_pct` (float, optional): Stop-loss percentage as decimal
- `take_profit_pct` (float, optional): Take-profit percentage as decimal
- `transaction_cost` (float, optional): Fixed transaction fee per trade
- `slippage` (float, optional): Additional cost per trade to model slippage
- `portfolio` (PortfolioManager, optional): Portfolio manager required when `df` is a dictionary

**Returns:**
- `pd.DataFrame` or `dict[str, pd.DataFrame]`: Trade results with strategy_return and equity_curve columns. When multiple symbols are provided, a `"portfolio"` key contains the aggregated results.

**Example:**
```python
from quanttradeai import simulate_trades

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
from quanttradeai import compute_metrics

# Calculate performance metrics
metrics = compute_metrics(df_with_trades, risk_free_rate=0.02)
print(f"Cumulative Return: {metrics['cumulative_return']:.4f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
```

## Complete Backtesting Workflow

### Basic Backtesting
```python
from quanttradeai import simulate_trades, compute_metrics

# Simulate trades
df_trades = simulate_trades(df_labeled)

# Calculate metrics
metrics = compute_metrics(df_trades)

print(f"Total Return: {metrics['cumulative_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

### Backtesting with Risk Management
```python
from quanttradeai import simulate_trades, compute_metrics, apply_stop_loss_take_profit

# Apply risk management rules
df_with_risk = apply_stop_loss_take_profit(
    df_labeled, 
    stop_loss_pct=0.02, 
    take_profit_pct=0.04
)

# Simulate trades
df_trades = simulate_trades(df_with_risk)

# Calculate metrics
metrics = compute_metrics(df_trades, risk_free_rate=0.02)

print(f"Risk-Adjusted Return: {metrics['cumulative_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

### Multi-Asset Backtesting
```python
from quanttradeai import simulate_trades, compute_metrics

# Backtest multiple assets
results = {}
for symbol, df in data_dict.items():
    # Process data and generate labels
    df_processed = processor.process_data(df)
    df_labeled = processor.generate_labels(df_processed)
    
    # Simulate trades
    df_trades = simulate_trades(df_labeled)
    
    # Calculate metrics
    metrics = compute_metrics(df_trades)
    results[symbol] = metrics

# Compare results
for symbol, metrics in results.items():
    print(f"{symbol}: Return={metrics['cumulative_return']:.2%}, "
          f"Sharpe={metrics['sharpe_ratio']:.2f}, "
          f"MDD={metrics['max_drawdown']:.2%}")
```

## Performance Analysis

### Equity Curve Analysis
```python
import matplotlib.pyplot as plt

# Simulate trades
df_trades = simulate_trades(df_labeled)

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.plot(df_trades.index, df_trades['equity_curve'])
plt.title('Strategy Equity Curve')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.show()

# Calculate drawdown
cumulative_max = df_trades['equity_curve'].cummax()
drawdown = (df_trades['equity_curve'] - cumulative_max) / cumulative_max

plt.figure(figsize=(12, 6))
plt.fill_between(df_trades.index, drawdown, 0, alpha=0.3, color='red')
plt.title('Strategy Drawdown')
plt.xlabel('Date')
plt.ylabel('Drawdown %')
plt.grid(True)
plt.show()
```

### Trade Analysis
```python
# Analyze individual trades
df_trades = simulate_trades(df_labeled)

# Calculate trade statistics
trades = df_trades[df_trades['strategy_return'] != 0]
winning_trades = trades[trades['strategy_return'] > 0]
losing_trades = trades[trades['strategy_return'] < 0]

print(f"Total Trades: {len(trades)}")
print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
print(f"Average Win: {winning_trades['strategy_return'].mean():.2%}")
print(f"Average Loss: {losing_trades['strategy_return'].mean():.2%}")
```

## Risk Management Integration

### Stop-Loss and Take-Profit
```python
from quanttradeai import apply_stop_loss_take_profit

# Apply different risk management scenarios
scenarios = [
    (None, None),           # No risk management
    (0.02, None),          # 2% stop-loss only
    (None, 0.04),          # 4% take-profit only
    (0.02, 0.04),         # Both stop-loss and take-profit
]

for sl, tp in scenarios:
    df_with_risk = apply_stop_loss_take_profit(df_labeled, sl, tp)
    df_trades = simulate_trades(df_with_risk)
    metrics = compute_metrics(df_trades)
    
    print(f"SL: {sl}, TP: {tp}")
    print(f"  Return: {metrics['cumulative_return']:.2%}")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  MDD: {metrics['max_drawdown']:.2%}")
```

## Configuration

### Backtesting Configuration
```yaml
trading:
  position_size: 0.2
  stop_loss: 0.02
  take_profit: 0.04
  max_positions: 5
  transaction_cost: 0.001
```

## Error Handling

### Data Validation
```python
# Check required columns
required_cols = ['Close', 'label']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Check label values
valid_labels = [-1, 0, 1]
invalid_labels = df['label'].unique()
invalid_labels = [l for l in invalid_labels if l not in valid_labels]
if invalid_labels:
    raise ValueError(f"Invalid label values: {invalid_labels}")
```

### Performance Issues
```python
try:
    # Simulate trades
    df_trades = simulate_trades(df_labeled)
    metrics = compute_metrics(df_trades)
except Exception as e:
    print(f"Backtesting error: {e}")
    # Check data quality
    print(f"Data shape: {df_labeled.shape}")
    print(f"Label distribution: {df_labeled['label'].value_counts()}")
```

## Performance Tips

### Memory Optimization
```python
# Use smaller date ranges for testing
df_sample = df_labeled['2023-01-01':'2023-12-31']
df_trades = simulate_trades(df_sample)
```

### Parallel Processing
```python
# Backtest multiple assets in parallel
from concurrent.futures import ProcessPoolExecutor

def backtest_asset(symbol_data):
    symbol, df = symbol_data
    df_trades = simulate_trades(df)
    return symbol, compute_metrics(df_trades)

with ProcessPoolExecutor() as executor:
    results = list(executor.map(backtest_asset, data_dict.items()))
```

## Related Documentation

- **[Data Loading](data.md)** - Data fetching and processing
- **[Feature Engineering](features.md)** - Technical indicators and features
- **[Machine Learning](models.md)** - Model training and evaluation
- **[Trading Utilities](trading.md)** - Risk management and position sizing
- **[Configuration](../configuration.md)** - Configuration guide
- **[Quick Reference](../quick-reference.md)** - Common patterns