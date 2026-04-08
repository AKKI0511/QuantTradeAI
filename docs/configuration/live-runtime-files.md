# Runtime And Live Trading Configs

`config/project.yaml` is now the canonical entrypoint for:

- research runs
- project-defined agent backtests
- project-defined agent paper runs
- agent backtest-to-paper promotion
- project-agent docker-compose deployment bundles

The legacy runtime YAML files still matter for:

- `train`
- `evaluate`
- `backtest-model`
- `live-trade`

This page explains that boundary.

## Runtime Files At A Glance

| File | Used by | What it controls |
| --- | --- | --- |
| `config/model_config.yaml` | `train`, `evaluate`, `backtest-model`, `live-trade` | Historical data settings, labels, training settings, trading defaults |
| `config/features_config.yaml` | `train`, `evaluate`, `backtest-model`, `live-trade` | Feature generation and preprocessing |
| `config/backtest_config.yaml` | `backtest`, `backtest-model` | Execution costs, slippage, liquidity, impact, borrow fees, intrabar |
| `config/impact_config.yaml` | `backtest-model`, validation | Per-asset-class impact defaults |
| `config/risk_config.yaml` | `backtest-model`, `live-trade` | Drawdown and turnover guards |
| `config/streaming.yaml` | `live-trade`, `StreamingGateway` | Streaming providers, subscriptions, reconnects, health endpoints |
| `config/position_manager.yaml` | `live-trade` | Position manager settings for the legacy live path |

## Important Boundary

### Canonical Paper Agent Path

`quanttradeai agent run --mode paper` for project-defined agents:

- reads `config/project.yaml`
- compiles runtime model, features, backtest, and streaming YAMLs into the run directory
- passes the compiled feature config into the paper runtime

### Legacy Live Path

`quanttradeai live-trade`:

- does **not** read `config/project.yaml`
- still reads `config/model_config.yaml` and `config/streaming.yaml`
- still uses the legacy runtime-YAML contract directly

## Legacy Live Trading Command

```bash
poetry run quanttradeai live-trade \
  -m models/experiments/<timestamp>/<SYMBOL> \
  -c config/model_config.yaml \
  -s config/streaming.yaml \
  --risk-config config/risk_config.yaml \
  --position-manager-config config/position_manager.yaml \
  --initial-capital 1000000 \
  --history-window 512 \
  --min-history 220 \
  --stop-loss-pct 0.01
```

## `config/model_config.yaml`

This file still drives the legacy saved-model and live-trading path.

Most important blocks:

- `data`
- `news`
- `training`
- `trading`

Example:

```yaml
data:
  symbols: ["AAPL", "META", "TSLA"]
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  timeframe: "1d"
  cache_dir: "data/raw"
  cache_path: "data/raw"
  cache_expiration_days: 7
  use_cache: true
  refresh: false
  max_workers: 1
  test_start: "2024-10-01"
  test_end: "2024-12-31"

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

trading:
  initial_capital: 100000
  stop_loss: 0.02
  take_profit: 0.04
  max_risk_per_trade: 0.02
  max_portfolio_risk: 0.10
```

## `config/features_config.yaml`

This remains the direct config surface for `DataProcessor` in the legacy path.

The canonical project-agent paper path no longer depends on the default file location because it passes a compiled runtime features config directly into the engine.

## `config/backtest_config.yaml`

Used by standalone `backtest` and `backtest-model`.

Execution settings are still resolved in this order for `backtest-model`:

1. asset-class defaults from `config/impact_config.yaml`
2. values from `config/backtest_config.yaml`
3. CLI overrides such as `--cost-bps`

## `config/risk_config.yaml`

Used by:

- `backtest-model`
- `live-trade`

Behavior difference:

- `backtest-model` continues without the guard if the file is missing
- `live-trade` fails fast if the configured risk file is missing

## `config/streaming.yaml`

Used by:

- `live-trade`
- direct `StreamingGateway` usage

The project-agent paper path generates an equivalent runtime YAML from `data.streaming` inside `project.yaml`, but that compiled file is written into the run directory and is not intended to replace your checked-in legacy `config/streaming.yaml`.

## `config/position_manager.yaml`

Used only by the legacy `live-trade` path.

The current project-defined `model` paper-agent workflow runs without a separate position-manager YAML and relies on the engine's portfolio state plus run artifacts for the happy path.

## Validation

Validate the legacy runtime YAML bundle:

```bash
poetry run quanttradeai validate-config --output-dir reports/config_validation
```

Validate the canonical project config:

```bash
poetry run quanttradeai validate -c config/project.yaml
```
