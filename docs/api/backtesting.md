# Backtesting API

## Overview

The backtesting API simulates label-driven strategies, computes gross and net metrics, and models execution effects such as transaction costs, slippage, liquidity, borrow fees, intrabar fills, and market impact.

## Public Imports

```python
from quanttradeai import (
    simulate_trades,
    compute_metrics,
    BacktestEngine,
    MarketImpactModel,
    LinearImpactModel,
    SquareRootImpactModel,
    AlmgrenChrissModel,
    ImpactCalculator,
    DynamicSpreadModel,
)

from quanttradeai.backtest import simulate_trades, compute_metrics, BacktestEngine
from quanttradeai.backtest.impact import ImpactCalculator, LinearImpactModel
```

## Main Classes and Functions

| API | Import Path | Purpose |
| --- | --- | --- |
| `simulate_trades` | `quanttradeai.backtest.backtester` | Simulate strategy returns from `label` signals |
| `compute_metrics` | `quanttradeai.backtest.backtester` | Compute gross/net performance metrics |
| `BacktestEngine` | `quanttradeai.backtest.engine` | Thin wrapper around `simulate_trades` with portfolio/risk wiring |
| `MarketImpactModel` | `quanttradeai.backtest.impact` | Protocol for impact model implementations |
| `LinearImpactModel` | `quanttradeai.backtest.impact` | Temporary square-root and permanent linear impact |
| `SquareRootImpactModel` | `quanttradeai.backtest.impact` | Temporary and permanent square-root impact |
| `AlmgrenChrissModel` | `quanttradeai.backtest.impact` | Basic Almgren-Chriss style impact model |
| `ImpactCalculator` | `quanttradeai.backtest.impact` | Applies impact model, spread, asymmetry, and decay |
| `DynamicSpreadModel` | `quanttradeai.backtest.impact` | Computes adaptive spread from volatility, volume, and time of day |

## `simulate_trades`

**Signature**

```python
def simulate_trades(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    execution: dict | None = None,
    portfolio: PortfolioManager | None = None,
    drawdown_guard: DrawdownGuard | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]
```

`simulate_trades` expects a `label` column where `1` means long, `-1` means short, and `0` means flat. It computes one-period-ahead strategy returns based on the current position.

| Input | Required | Notes |
| --- | --- | --- |
| `Close` | Yes | Used as the default reference price |
| `label` | Yes | Signal/target direction |
| `Volume` | Optional | Used for liquidity and impact; defaults to effectively unlimited volume |
| `Mid` | Optional | Used when slippage reference price is `mid` |
| `order_type` | Optional | `market`, `limit`, or `stop`; defaults to `market` |
| `limit_price` | Optional | Used for limit orders |
| `stop_price` | Optional | Used for stop orders |
| `Volatility` | Optional | Used by dynamic spread/impact; defaults to `0.0` |
| `ticks` | Optional | Used by intrabar execution when configured |

**Returns**

For a single DataFrame, returns a copy with:

| Column or Attr | Meaning |
| --- | --- |
| `gross_return` | Strategy return before execution costs |
| `strategy_return` | Strategy return after execution costs |
| `gross_equity_curve` | Cumulative gross equity |
| `equity_curve` | Cumulative net equity |
| `data.attrs["ledger"]` | Per-fill execution ledger as a DataFrame |

For a `dict[str, DataFrame]`, returns per-symbol results plus a `"portfolio"` DataFrame with aggregated `strategy_return` and `equity_curve`. Multi-symbol simulation requires a `PortfolioManager`.

```python
from quanttradeai import simulate_trades, compute_metrics

results = simulate_trades(
    labeled_bars,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    execution={
        "transaction_costs": {"enabled": True, "mode": "bps", "value": 1.0},
        "slippage": {"enabled": True, "mode": "bps", "value": 2.0},
    },
)

metrics = compute_metrics(results)
ledger = results.attrs["ledger"]
```

### Execution Config

`execution` is a nested dictionary. The simulator currently reads these sections:

| Section | Key Behavior |
| --- | --- |
| `transaction_costs` | Adds bps, fixed-notional, or per-share costs |
| `slippage` | Adds bps or fixed price slippage |
| `liquidity` | Caps fills by participation and optional order book depth |
| `impact` | Enables `ImpactCalculator` and optional dynamic spread |
| `borrow_fee` | Adds fee rows while short |
| `intrabar` | Uses `ticks` column or synthetic ticks for fills |

The scalar `transaction_cost` and `slippage` arguments are shortcuts that are converted to bps execution config values.

## `compute_metrics`

**Signature**

```python
def compute_metrics(data: pd.DataFrame, risk_free_rate: float = 0.0) -> dict
```

Delegates to `quanttradeai.utils.metrics.compute_performance`. It expects `strategy_return` and `equity_curve`; when present, `gross_return` and `gross_equity_curve` are used for gross metrics.

**Returns**

```python
{
    "gross_pnl": float,
    "total_costs": float,
    "total_slippage_cost": float,
    "total_impact_cost": float,
    "net_pnl": float,
    "gross_sharpe": float,
    "net_sharpe": float,
    "gross_cagr": float,
    "net_cagr": float,
    "gross_mdd": float,
    "net_mdd": float,
    "cumulative_return": float,
    "sharpe_ratio": float,
    "max_drawdown": float,
}
```

```python
metrics = compute_metrics(results, risk_free_rate=0.03)
```

## `BacktestEngine`

**Signature**

```python
@dataclass
class BacktestEngine:
    portfolio: PortfolioManager | None = None
    risk_manager: RiskManager | None = None

    def run(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        execution: dict | None = None,
        **kwargs,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]
```

`BacktestEngine` forwards to `simulate_trades`. If both `portfolio` and `risk_manager` are provided, the portfolio is wired to use the risk manager and the risk manager's drawdown guard is passed into the simulation.

```python
from quanttradeai import BacktestEngine
from quanttradeai.trading import PortfolioManager, RiskManager

engine = BacktestEngine(
    portfolio=PortfolioManager(capital=100_000),
    risk_manager=RiskManager(),
)

results = engine.run(labeled_bars)
```

## Market Impact Models

### `MarketImpactModel`

**Signature**

```python
class MarketImpactModel(Protocol):
    alpha: float
    beta: float
    def temporary(self, trade_volume: float, adv: float) -> float: ...
    def permanent(self, trade_volume: float, adv: float) -> float: ...
```

Implement this protocol when adding custom impact models. `adv` is average daily volume or another volume baseline chosen by the caller.

### `LinearImpactModel`

**Signature**

```python
@dataclass
class LinearImpactModel:
    alpha: float = 0.0
    beta: float = 0.0
```

Behavior:

- `temporary(trade_volume, adv)`: `alpha * sqrt(trade_volume / adv)`
- `permanent(trade_volume, adv)`: `beta * (trade_volume / adv)`
- If `adv <= 0`, the ratio is treated as `0.0`.

### `SquareRootImpactModel`

**Signature**

```python
@dataclass
class SquareRootImpactModel(LinearImpactModel):
    alpha: float = 0.0
    beta: float = 0.0
```

Behavior:

- `temporary(...)`: `alpha * sqrt(trade_volume / adv)`
- `permanent(...)`: `beta * sqrt(trade_volume / adv)`

### `AlmgrenChrissModel`

**Signature**

```python
@dataclass
class AlmgrenChrissModel(LinearImpactModel):
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
```

Behavior:

- `temporary(...)`: `alpha * ratio + beta * sqrt(ratio)`
- `permanent(...)`: `gamma * ratio`

### `DynamicSpreadModel`

**Signature**

```python
class DynamicSpreadModel:
    def __init__(
        self,
        base: float = 0.0,
        vol_coeff: float = 0.0,
        volume_coeff: float = 0.0,
        tod: dict[int, float] | None = None,
    ) -> None

    def value(self, volatility: float, volume: float, ts: datetime) -> float
```

Computes:

```text
base + vol_coeff * volatility + volume_coeff / volume
```

The volume term is applied only when `volume > 0`. The result is then multiplied by an optional time-of-day multiplier from `tod` by `ts.hour`.

### `ImpactCalculator`

**Signature**

```python
@dataclass
class ImpactCalculator:
    model: MarketImpactModel
    decay: float = 0.0
    decay_volume_coeff: float = 0.0
    spread: float = 0.0
    spread_model: DynamicSpreadModel | None = None
    alpha_buy: float | None = None
    alpha_sell: float | None = None
    beta_buy: float | None = None
    beta_sell: float | None = None
    cross_alpha: float = 0.0
    cross_beta: float = 0.0
    horizon_decay: float = 0.0

    def impact_per_share(...) -> tuple[float, float, float]
    def impact_cost(...) -> dict[str, float]
```

`impact_per_share` and `impact_cost` accept:

```python
trade_volume: float
adv: float
side: int = 1
volatility: float = 0.0
volume: float = 0.0
timestamp: datetime | None = None
cross_volume: float = 0.0
```

`impact_cost` returns:

```python
{
    "temp": float,
    "perm": float,
    "spread": float,
    "total": float,
}
```

```python
from quanttradeai import ImpactCalculator, LinearImpactModel

impact = ImpactCalculator(
    model=LinearImpactModel(alpha=0.1, beta=0.02),
    spread=0.01,
)

cost = impact.impact_cost(trade_volume=10_000, adv=1_000_000, side=1)
```

## Minimal Examples

### Single Symbol

```python
from quanttradeai import simulate_trades, compute_metrics

prepared = features[["Close", "Volume", "label"]].copy()
results = simulate_trades(prepared)
metrics = compute_metrics(results)
```

### Multi-Symbol Portfolio

```python
from quanttradeai import simulate_trades
from quanttradeai.trading import PortfolioManager

portfolio = PortfolioManager(capital=100_000)
results = simulate_trades(
    {"AAPL": aapl_bars, "MSFT": msft_bars},
    portfolio=portfolio,
)

portfolio_curve = results["portfolio"]["equity_curve"]
```

## Related CLI/YAML Docs

- [CLI docs](../cli/)
- [Config docs](../config/)
- [Artifacts](../artifacts.md)
- [Execution cost examples](../examples/execution-costs.md)

The research and agent backtest CLI paths call the same simulation and metrics functions after preparing labeled data.

## Common Mistakes

- Missing `label` or `Close` columns before calling `simulate_trades`.
- Passing a multi-symbol dictionary without a `PortfolioManager`.
- Comparing `gross_sharpe` to `net_sharpe` without checking `total_costs`, `total_slippage_cost`, and `total_impact_cost`.
- Assuming scalar `transaction_cost` and `slippage` are raw dollars; they are decimal fractions converted to bps.
- Enabling liquidity or impact without realistic `Volume` or `average_daily_volume` values.
- Reading only `equity_curve` and ignoring the execution ledger stored in `result.attrs["ledger"]`.
