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

The `model-agent` template also creates a placeholder model artifact at `models/trained/aapl_daily_classifier/README.md`. Replace that directory with a real saved model before you run the agent.

### LLM And Hybrid Agents

```bash
poetry run quanttradeai init --template llm-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode paper
```

The `llm-agent` and `hybrid` templates also create starter prompt files under `prompts/` so validation can pass immediately.

Current support:

| Agent kind | Backtest | Paper | Live |
| --- | --- | --- | --- |
| `model` | Yes | Yes | No |
| `llm` | Yes | Yes | No |
| `hybrid` | Yes | Yes | No |
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
  symbols:
    - ticker: "AAPL"
      asset_class: "equities"
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
    health_check_interval: 30

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

`symbols` may be either:

- plain ticker strings such as `["AAPL", "MSFT"]`
- mappings with `ticker` and optional `asset_class`

Example:

```yaml
data:
  symbols:
    - ticker: "AAPL"
      asset_class: "equities"
    - ticker: "MSFT"
      asset_class: "equities"
```

When you use mapping syntax, validation checks `asset_class` against the asset classes defined in `config/impact_config.yaml`.

`test_start` and `test_end` must fall inside the configured `start_date` and `end_date` range.

### `data.streaming`

Used by project-defined paper agents.

Required when any agent is configured with `mode: paper`:

- `enabled: true`
- `provider`
- `websocket_url`
- `auth_method`
- `symbols`
- `channels`
- `buffer_size`
- `reconnect_attempts`
- `health_check_interval`

Optional passthrough sections:

- `monitoring`
- `thresholds`
- `alerts`
- `metrics`
- `api`
- `rate_limit`
- `circuit_breaker`

These values are compiled into the runtime streaming YAML consumed by the current gateway.

Validation also enforces one important rule here: if any agent is configured with `mode: paper`, then `data.streaming.enabled` must be `true`.

### `features`

Canonical projects define reusable features in `features.definitions`.

Supported feature families in the canonical compiler:

- `technical`
- `custom`

If you define at least one `technical` feature, QuantTradeAI compiles the standard technical pipeline and defaults RSI to `14` when you do not provide a period explicitly.

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

Practical notes:

- `evaluation.use_configured_test_window: true` keeps `data.test_start` and `data.test_end` in the compiled runtime model config
- setting it to `false` clears those fields and lets the downstream research runtime fall back to its chronological split behavior
- `model.tuning.enabled` and `model.tuning.trials` flow directly into the training run
- `backtest.costs.bps` becomes the transaction cost setting in the compiled runtime backtest config

Even when `research.enabled` is `false`, the agent runtime still reuses these defaults to compile consistent runtime configs.

### `agents`

#### `model` agents

Required fields:

- `name`
- `kind: model`
- `model.path`

Validation checks that `model.path` exists and is not blank.

Backtest mode:

- loads the saved model artifact
- generates deterministic decision records
- writes `summary.json`, `metrics.json`, `equity_curve.csv`, `decisions.jsonl`, and `ledger.csv` when trades are present
- also writes compiled runtime config files and per-symbol coverage and metrics files under the run directory

Paper mode:

- compiles `data.streaming` into a runtime streaming config
- runs the existing paper/live engine with the compiled feature config
- writes `summary.json`, `metrics.json`, `executions.jsonl`, and `runtime_streaming_config.yaml`

#### `llm` and `hybrid` agents

`llm` and `hybrid` require an `llm` block with:

- `provider`
- `model`
- `prompt_file`

Validation checks that `prompt_file` exists relative to the project config location.

Hybrid agents may also define `model_signal_sources`.

`model_signal_sources` must be written as objects with `name` and `path` for runnable agent configs. Legacy string entries can still pass `quanttradeai validate` with a deprecation warning, but `quanttradeai agent run --mode backtest` raises a runtime `ValueError` when loading them.

Paper mode:

- compiles `data.streaming` into a runtime streaming config
- warm-starts with recent historical OHLCV before streaming begins
- aggregates streaming messages into completed bars before invoking the agent
- writes `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, `prompt_samples.json`, and `runtime_streaming_config.yaml`

### `deployment`

Required metadata. It is not yet a complete workflow driver.

## Runtime Artifacts

### `quanttradeai validate`

Writes resolved config artifacts under `reports/config_validation/<timestamp>/`.

You will get:

- `resolved_project_config.yaml`
- `summary.json`

When you validate with `--legacy-config-dir`, QuantTradeAI also writes `migrated_project_config.yaml`.

### `quanttradeai research run`

Writes under `runs/research/<timestamp>_<project>/`.

Alongside `summary.json` and `metrics.json`, QuantTradeAI writes:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `runtime_backtest_config.yaml`
- `backtest_summary.json` when automatic post-train backtests produce output

### `quanttradeai agent run --mode backtest`

Writes under `runs/agent/backtest/<timestamp>_<agent>/`.

### `quanttradeai agent run --mode paper`

Writes under `runs/agent/paper/<timestamp>_<agent>/`.

For `llm` and `hybrid` paper runs, the run directory includes both `decisions.jsonl` and `executions.jsonl` so prompt-driven decisions and actual paper executions can be audited separately.

## Compatibility Notes

The canonical compiler still accepts some compatibility sections while the repo transitions from legacy YAMLs:

- `news`
- `training`
- `trading`
- `execution`
- `models`

Treat those as migration aids, not the preferred long-term shape.
