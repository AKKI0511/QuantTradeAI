# Trading API

## Overview

The trading API provides portfolio accounting, position sizing, stop-loss/take-profit label adjustment, and live/paper risk controls. Only `PortfolioManager`, `apply_stop_loss_take_profit`, and `position_size` are top-level lazy exports from `quanttradeai`; the position and risk managers are available from `quanttradeai.trading`.

## Public Imports

```python
from quanttradeai import PortfolioManager, apply_stop_loss_take_profit, position_size

from quanttradeai.trading import (
    PortfolioManager,
    PositionManager,
    DrawdownGuard,
    RiskManager,
    apply_stop_loss_take_profit,
    position_size,
)
```

## Main Classes and Functions

| API | Import Path | Purpose |
| --- | --- | --- |
| `PortfolioManager` | `quanttradeai.trading.portfolio` | Cash, positions, realized PnL, and risk-aware allocation |
| `apply_stop_loss_take_profit` | `quanttradeai.trading.risk` | Adjust `label` signals after stop/take-profit triggers |
| `position_size` | `quanttradeai.trading.risk` | Calculate integer quantity from account risk |
| `PositionManager` | `quanttradeai.trading.position_manager` | Thread-safe live/paper position state and execution analytics |
| `DrawdownGuard` | `quanttradeai.trading.drawdown_guard` | Drawdown and turnover protection |
| `RiskManager` | `quanttradeai.trading.risk_manager` | Thin coordinator around risk guards |

## `PortfolioManager`

**Signature**

```python
class PortfolioManager:
    def __init__(
        self,
        capital: float,
        max_risk_per_trade: float = 0.02,
        max_portfolio_risk: float = 0.1,
        risk_manager: RiskManager | None = None,
        drawdown_guard: DrawdownGuard | None = None,
    ) -> None
```

`PortfolioManager` tracks cash, open long positions, realized PnL, and portfolio risk exposure.

| Property or Method | Signature | Return |
| --- | --- | --- |
| `portfolio_value` | property | `cash + market value of positions` |
| `risk_exposure` | property | Portfolio risk fraction |
| `estimate_open_position_qty` | `(price, stop_loss_pct=None, *, check_risk=True)` | Quantity that would be allocated |
| `open_position` | `(symbol, price, stop_loss_pct=None)` | Quantity opened |
| `close_position` | `(symbol, price)` | Quantity closed |
| `close_all_positions` | `(prices=None)` | `dict[str, int]` closed by symbol |
| `replace_state` | `(*, cash, positions, initial_capital=None, realized_pnl=None)` | `None` |

```python
from quanttradeai import PortfolioManager

portfolio = PortfolioManager(
    capital=100_000,
    max_risk_per_trade=0.02,
    max_portfolio_risk=0.10,
)

qty = portfolio.open_position("AAPL", price=180.0, stop_loss_pct=0.02)
closed = portfolio.close_position("AAPL", price=185.0)
```

**Errors and edge cases**

- `open_position` raises `ValueError` if the symbol already has a position.
- `open_position` returns `0` when risk guards block trading or cash is insufficient.
- `drawdown_guard` and `risk_manager` are mutually exclusive constructor arguments.
- Positions are represented internally as dictionaries with `qty`, `price`, `entry_price`, and `stop_loss_pct`.

## `apply_stop_loss_take_profit`

**Signature**

```python
def apply_stop_loss_take_profit(
    df: pd.DataFrame,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> pd.DataFrame
```

Adjusts the `label` column based on `Close` price movement from the active entry price.

| Input Column | Required |
| --- | --- |
| `Close` | Yes |
| `label` | Yes |

```python
from quanttradeai import apply_stop_loss_take_profit

adjusted = apply_stop_loss_take_profit(
    labeled_bars,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
)
```

**Return**

Returns a copy of the input DataFrame with an adjusted `label` column.

## `position_size`

**Signature**

```python
def position_size(
    capital: float,
    risk_per_trade: float,
    stop_loss_pct: float,
    price: float,
) -> int
```

Calculates:

```text
quantity = capital * risk_per_trade / (price * stop_loss_pct)
```

Then returns `max(int(quantity), 0)`.

```python
from quanttradeai import position_size

qty = position_size(
    capital=100_000,
    risk_per_trade=0.02,
    stop_loss_pct=0.05,
    price=50.0,
)
```

**Errors**

- Raises `ValueError` if `price <= 0`.
- Raises `ValueError` if `stop_loss_pct <= 0`.

## `PositionManager`

**Signature**

```python
@dataclass
class PositionManager:
    risk_manager: RiskManager | None = None
    impact: ImpactCalculator | None = None
    reconciliation: dict[str, str] = {"intraday": "1m", "daily": "1d"}
    mode: str = "paper"
    cash: float = 0.0

    @classmethod
    def from_config(cls, config: PositionManagerConfig | dict | str | None = None) -> PositionManager
```

`PositionManager` is the live/paper state manager used around streaming execution. It is thread-safe and can bind to a streaming gateway.

| Method | Signature | Return |
| --- | --- | --- |
| `from_config` | `(config=None)` | `PositionManager` |
| `bind_gateway` | `(gateway, symbols)` | `None` |
| `handle_market_data` | `(message)` | `None` |
| `open_position` | `(symbol, qty, price, adv=None, timestamp=None)` | `None` |
| `close_position` | `(symbol, price, adv=None, timestamp=None)` | Quantity closed |
| `portfolio_value` | property | Cash plus market value |
| `reconcile_positions` | `(now=None)` | `{"intraday": ..., "daily": ...}` |
| `execution_metrics` | `()` | `{"trades": int, "total_impact_cost": float}` |
| `replace_state` | `(*, cash, positions)` | `None` |

```python
from quanttradeai.trading import PositionManager

positions = PositionManager.from_config("config/position_manager.yaml")
positions.open_position("AAPL", qty=10, price=180.0)
metrics = positions.execution_metrics()
```

**Expected market data message**

`handle_market_data` reads:

| Field | Notes |
| --- | --- |
| `symbol` | Required to update a position |
| `price` or `last` or `close` | Required price source |
| `timestamp` | Optional; defaults to current UTC time |

## `DrawdownGuard`

**Signature**

```python
class DrawdownGuard:
    def __init__(
        self,
        config: DrawdownProtectionConfig | RiskManagementConfig | dict | None = None,
        turnover_limits: TurnoverLimitsConfig | None = None,
        config_path: str | None = None,
    ) -> None
```

Monitors portfolio values and trade notional to update risk state.

| Method | Return |
| --- | --- |
| `update_portfolio_value(current_value, timestamp)` | `None` |
| `record_trade(notional, timestamp)` | `None` |
| `check_drawdown_limits()` | Status dictionary |
| `get_position_size_multiplier()` | `float` |
| `should_halt_trading()` | `bool` |
| `should_emergency_liquidate()` | `bool` |
| `get_risk_metrics()` | Metrics dictionary |
| `reset_high_water_mark()` | `None` |

```python
from datetime import datetime, UTC

from quanttradeai.trading import DrawdownGuard

guard = DrawdownGuard({"max_drawdown_pct": 0.10})
guard.update_portfolio_value(100_000, datetime.now(UTC))
guard.update_portfolio_value(92_000, datetime.now(UTC))
status = guard.check_drawdown_limits()
```

## `RiskManager`

**Signature**

```python
class RiskManager:
    def __init__(self, drawdown_guard: DrawdownGuard | None = None) -> None
```

Coordinates one optional `DrawdownGuard` behind a small interface used by `PortfolioManager`, `PositionManager`, `BacktestEngine`, and live trading.

```python
from quanttradeai.trading import DrawdownGuard, RiskManager

risk = RiskManager(drawdown_guard=DrawdownGuard({"max_drawdown_pct": 0.10}))
```

## Minimal Examples

### Risk-Aware Portfolio Allocation

```python
from quanttradeai import PortfolioManager

portfolio = PortfolioManager(capital=250_000, max_risk_per_trade=0.01)
qty = portfolio.estimate_open_position_qty(price=125.0, stop_loss_pct=0.03)
```

### Use in Multi-Symbol Backtest

```python
from quanttradeai import simulate_trades
from quanttradeai.trading import PortfolioManager

portfolio = PortfolioManager(capital=100_000)
results = simulate_trades(
    {"AAPL": aapl_labeled, "MSFT": msft_labeled},
    portfolio=portfolio,
)
```

## Related CLI/YAML Docs

- [CLI docs](../cli/)
- [Risk and position config](../config/risk-position-deployment.md)
- [Artifacts](../artifacts.md)

Risk and position settings in YAML are compiled into the same manager classes for backtest, paper, and live agent runs.

## Common Mistakes

- Passing both `risk_manager` and `drawdown_guard` to `PortfolioManager`.
- Calling `position_size` with `stop_loss_pct=0`; use `PortfolioManager.estimate_open_position_qty` if you want allocation without a stop distance.
- Expecting `apply_stop_loss_take_profit` to execute trades; it only adjusts labels before simulation.
- Supplying short positions to `PortfolioManager`; it tracks long positions for portfolio allocation, while `simulate_trades` can simulate short labels internally.
- Forgetting that `PositionManager.open_position` mutates cash and execution records immediately.
