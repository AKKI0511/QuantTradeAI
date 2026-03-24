# Project Config (`project.yaml`)

`config/project.yaml` is the canonical config entrypoint for QuantTradeAI.

It drives:

- `quanttradeai init`
- `quanttradeai validate`
- `quanttradeai research run`
- `quanttradeai agent run` for project-defined agents

`live-trade` still uses the legacy runtime YAML files documented in [Runtime and Live Trading Configs](live-runtime-files.md).

## Supported Project Workflows

### Research

```bash
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
```

### Model Agents

```bash
poetry run quanttradeai init --template model-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode paper
```

### LLM And Hybrid Agents

```bash
poetry run quanttradeai init --template llm-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
```

Current support:

| Agent kind | Backtest | Paper | Live |
| --- | --- | --- | --- |
| `model` | Yes | Yes | No |
| `llm` | Yes | No | No |
| `hybrid` | Yes | No | No |
| `rule` | No | No | No |

## Canonical Shape

```yaml
project:
  name: "intraday_lab"
  profile: "paper"

profiles:
  research:
    mode: "research"
  paper:
    mode: "paper"
  live:
    mode: "live"

data:
  symbols: ["AAPL"]
  start_date: "2022-01-01"
  end_date: "2024-12-31"
  timeframe: "1d"
  test_start: "2024-09-01"
  test_end: "2024-12-31"
  streaming:
    enabled: true
    provider: "alpaca"
    websocket_url: "wss://stream.data.alpaca.markets/v2/iex"
    auth_method: "api_key"
    symbols: ["AAPL"]
    channels: ["trades", "quotes"]
    buffer_size: 1000
    reconnect_attempts: 5

features:
  definitions:
    - name: "rsi_14"
      type: "technical"
      params: { period: 14 }

research:
  enabled: false
  labels:
    type: "forward_return"
    horizon: 5
    buy_threshold: 0.01
    sell_threshold: -0.01
  model:
    kind: "classifier"
    family: "voting"
    tuning: { enabled: false, trials: 1 }
  evaluation:
    split: "time_aware"
    use_configured_test_window: true
  backtest:
    costs: { enabled: true, bps: 5 }

agents:
  - name: "paper_momentum"
    kind: "model"
    mode: "paper"
    model:
      path: "models/trained/aapl_daily_classifier"
    context:
      features: ["rsi_14"]
      positions: true
      risk_state: true
    risk:
      max_position_pct: 0.05

deployment:
  target: "docker-compose"
  mode: "paper"
```

## Section Reference

### `project`

- `name`: used in summaries and run-directory naming
- `profile`: metadata surfaced in validation and run outputs

### `profiles`

Required metadata describing available environments. This is still mostly descriptive today.

### `data`

Core fields used by research and agent runtimes:

- `symbols`
- `start_date`
- `end_date`
- `timeframe`
- `test_start`
- `test_end`
- `cache_dir`
- `cache_path`
- `cache_expiration_days`
- `use_cache`
- `refresh`
- `max_workers`

`symbols` may be plain tickers or mappings with `ticker` and `asset_class`.

### `data.streaming`

Used by project-defined `model` agents in paper mode.

Required when a `model` agent is configured with `mode: paper`:

- `enabled: true`
- `provider`
- `websocket_url`
- `auth_method`
- `symbols`
- `channels`
- `buffer_size`
- `reconnect_attempts`

Optional passthrough sections:

- `monitoring`
- `thresholds`
- `alerts`
- `metrics`
- `api`
- `rate_limit`
- `circuit_breaker`

These values are compiled into the runtime streaming YAML consumed by the current gateway.

### `features`

Canonical projects define reusable features in `features.definitions`.

Supported feature families in the canonical compiler:

- `technical`
- `custom`

Supported custom feature kinds:

- `price_momentum`
- `volume_momentum`
- `mean_reversion`
- `volatility_breakout`

### `research`

The canonical research compiler reads:

- `labels`
- `model`
- `evaluation.use_configured_test_window`
- `backtest.costs`

Even when `research.enabled` is `false`, the agent runtime still reuses these defaults to compile consistent runtime configs.

### `agents`

#### `model` agents

Required fields:

- `name`
- `kind: model`
- `model.path`

Backtest mode:

- loads the saved model artifact
- generates deterministic decision records
- writes `summary.json`, `metrics.json`, `equity_curve.csv`, `ledger.csv` when present, and `decisions.jsonl`

Paper mode:

- compiles `data.streaming` into a runtime streaming config
- runs the existing paper/live engine with the compiled feature config
- writes `summary.json`, `metrics.json`, and `executions.jsonl`

#### `llm` and `hybrid` agents

Supported today in `backtest` mode only.

`llm` and `hybrid` require an `llm` block with:

- `provider`
- `model`
- `prompt_file`

Hybrid agents may also define `model_signal_sources`.

### `deployment`

Required metadata. It is not yet a complete workflow driver.

## Runtime Artifacts

### `quanttradeai validate`

Writes resolved config artifacts under `reports/config_validation/<timestamp>/`.

### `quanttradeai research run`

Writes under `runs/research/<timestamp>_<project>/`.

### `quanttradeai agent run --mode backtest`

Writes under `runs/agent/backtest/<timestamp>_<agent>/`.

### `quanttradeai agent run --mode paper`

Writes under `runs/agent/paper/<timestamp>_<agent>/`.

## Compatibility Notes

The canonical compiler still accepts some compatibility sections while the repo transitions from legacy YAMLs:

- `news`
- `training`
- `trading`
- `execution`
- `models`

Treat those as migration aids, not the preferred long-term shape.
