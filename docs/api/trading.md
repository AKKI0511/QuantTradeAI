# Trading Utilities

API documentation for risk management and position sizing.

## Risk Management

### `apply_stop_loss_take_profit(df: pd.DataFrame, stop_loss_pct: float | None = None, take_profit_pct: float | None = None) -> pd.DataFrame`

Applies stop-loss and take-profit rules to trading signals.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with Close prices and label column
- `stop_loss_pct` (float, optional): Stop-loss percentage as decimal
- `take_profit_pct` (float, optional): Take-profit percentage as decimal

**Returns:**
- `pd.DataFrame`: DataFrame with adjusted label column

**Example:**
```python
from quanttradeai import apply_stop_loss_take_profit

# Apply 2% stop-loss and 4% take-profit
df_with_risk = apply_stop_loss_take_profit(df, stop_loss_pct=0.02, take_profit_pct=0.04)

# Apply only stop-loss
df_with_sl = apply_stop_loss_take_profit(df, stop_loss_pct=0.03)

# Apply only take-profit
df_with_tp = apply_stop_loss_take_profit(df, take_profit_pct=0.05)
```

### `position_size(capital: float, risk_per_trade: float, stop_loss_pct: float, price: float) -> int`

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
from quanttradeai import position_size

# Calculate position size
qty = position_size(capital=10000, risk_per_trade=0.02, stop_loss_pct=0.05, price=150.0)
print(f"Position size: {qty} shares")
```

### `PortfolioManager(capital: float, max_risk_per_trade: float = 0.02, max_portfolio_risk: float = 0.1)`

Manages capital allocation and risk across multiple symbols.

**Parameters:**
- `capital` (float): Initial portfolio capital
- `max_risk_per_trade` (float, optional): Risk per trade as a fraction of portfolio value
- `max_portfolio_risk` (float, optional): Maximum overall portfolio risk exposure

**Example:**
```python
from quanttradeai import PortfolioManager

# Create portfolio manager with $10,000 starting capital
pm = PortfolioManager(10000, max_risk_per_trade=0.02, max_portfolio_risk=0.10)

# Open a position without a stop loss
qty = pm.open_position("AAPL", price=150)
print(f"AAPL position: {qty} shares")
```

## Risk Management Workflows

### Basic Risk Management
```python
from quanttradeai import apply_stop_loss_take_profit

# Apply risk management to trading signals
df_with_risk = apply_stop_loss_take_profit(
    df_labeled, 
    stop_loss_pct=0.02, 
    take_profit_pct=0.04
)

# Check risk-adjusted signals
risk_adjusted_signals = df_with_risk['label'].value_counts()
print(f"Risk-adjusted signals: {risk_adjusted_signals}")
```

### Dynamic Position Sizing
```python
from quanttradeai import position_size

# Calculate position sizes for different scenarios
scenarios = [
    (10000, 0.01, 0.05, 150.0),  # Conservative
    (10000, 0.02, 0.05, 150.0),  # Moderate
    (10000, 0.03, 0.05, 150.0),  # Aggressive
]

for capital, risk, sl, price in scenarios:
    qty = position_size(capital, risk, sl, price)
    print(f"Capital: ${capital}, Risk: {risk*100}%, Position: {qty} shares")
```

### Portfolio Risk Management
```python
from quanttradeai import PortfolioManager

# Manage risk across multiple positions using PortfolioManager
pm = PortfolioManager(50000, max_risk_per_trade=0.02, max_portfolio_risk=0.10)

# Open positions
pm.open_position('AAPL', price=150.0, stop_loss_pct=0.05)
pm.open_position('TSLA', price=250.0, stop_loss_pct=0.05)
pm.open_position('META', price=300.0, stop_loss_pct=0.05)

print(f"Current exposure: {pm.risk_exposure:.2%}")
```

## Risk Analysis

### Risk Metrics Calculation
```python
import pandas as pd
import numpy as np

def calculate_risk_metrics(df_trades):
    """Calculate comprehensive risk metrics."""
    returns = df_trades['strategy_return']
    
    # Basic metrics
    total_return = df_trades['equity_curve'].iloc[-1] - 1
    volatility = returns.std() * np.sqrt(252)  # Annualized
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    
    # Drawdown analysis
    equity_curve = df_trades['equity_curve']
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    # VaR (Value at Risk)
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'var_99': var_99
    }

# Calculate risk metrics
metrics = calculate_risk_metrics(df_trades)
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"VaR (95%): {metrics['var_95']:.2%}")
print(f"VaR (99%): {metrics['var_99']:.2%}")
```

### Risk-Adjusted Performance
```python
def risk_adjusted_analysis(df_trades, risk_free_rate=0.02):
    """Analyze risk-adjusted performance metrics."""
    returns = df_trades['strategy_return']
    
    # Risk-adjusted metrics
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252)
    
    # Calmar ratio
    equity_curve = df_trades['equity_curve']
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    max_drawdown = abs(drawdown.min())
    calmar_ratio = returns.mean() * 252 / max_drawdown
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio
    }

# Calculate risk-adjusted metrics
risk_metrics = risk_adjusted_analysis(df_trades)
print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
print(f"Calmar Ratio: {risk_metrics['calmar_ratio']:.2f}")
```

## Configuration

### Risk Management Configuration
```yaml
trading:
  position_size: 0.2
  stop_loss: 0.02
  take_profit: 0.04
  max_positions: 5
  transaction_cost: 0.001
  max_risk_per_trade: 0.02
  max_portfolio_risk: 0.10
```

## Error Handling

### Invalid Parameters
```python
try:
    # Apply risk management with valid parameters
    df_with_risk = apply_stop_loss_take_profit(df, stop_loss_pct=0.02)
except ValueError as e:
    print(f"Risk management error: {e}")
    # Check parameter validity
    if stop_loss_pct < 0 or stop_loss_pct > 1:
        print("Stop-loss must be between 0 and 1")
```

### Position Sizing Errors
```python
try:
    # Calculate position size
    qty = position_size(capital=10000, risk_per_trade=0.02, 
                       stop_loss_pct=0.05, price=150.0)
except ValueError as e:
    print(f"Position sizing error: {e}")
    # Check parameter validity
    if price <= 0:
        print("Price must be positive")
    if stop_loss_pct <= 0:
        print("Stop-loss must be positive")
```

## Performance Tips

### Efficient Risk Management
```python
# Apply risk management in batches for large datasets
def apply_risk_batch(df, batch_size=1000):
    """Apply risk management in batches."""
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_with_risk = apply_stop_loss_take_profit(batch, stop_loss_pct=0.02)
        results.append(batch_with_risk)
    return pd.concat(results)
```

### Memory Optimization
```python
# Use smaller data types for large datasets
df['label'] = df['label'].astype('int8')
df['Close'] = df['Close'].astype('float32')
```

## Related Documentation

- **[Data Loading](data.md)** - Data fetching and processing
- **[Feature Engineering](features.md)** - Technical indicators and features
- **[Machine Learning](models.md)** - Model training and evaluation
- **[Backtesting](backtesting.md)** - Trade simulation and evaluation
- **[Configuration](../configuration.md)** - Configuration guide
- **[Quick Reference](../quick-reference.md)** - Common patterns