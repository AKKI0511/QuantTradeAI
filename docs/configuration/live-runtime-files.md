# Runtime and Live Trading Configs

`config/project.yaml` is the canonical file for research and agent backtests.

Live trading and several execution-oriented workflows still use a separate runtime YAML set today. This page explains what each file controls and where the boundaries are.

## At a Glance

| File | Used by | What it controls |
| --- | --- | --- |
| `config/model_config.yaml` | `train`, `evaluate`, `backtest-model`, `live-trade` | Historical data settings, label settings, training settings, and portfolio sizing defaults |
| `config/features_config.yaml` | `train`, `evaluate`, `live-trade`, saved-model backtests | Feature generation, preprocessing, sentiment, and multi-timeframe feature config |
| `config/backtest_config.yaml` | `backtest`, `backtest-model` | Execution costs, slippage, liquidity, impact, borrow fees, intrabar settings |
| `config/impact_config.yaml` | `backtest-model`, validator | Per-asset-class impact defaults |
| `config/risk_config.yaml` | `backtest-model`, `live-trade` | Drawdown guard and turnover limits |
| `config/streaming.yaml` | `live-trade`, `StreamingGateway` | Streaming providers, subscriptions, buffer sizing, reconnects, health endpoints |
| `config/position_manager.yaml` | `live-trade` | Live position tracking, intraday risk config, reconciliation windows |

## Live Trading Command

The current live path is:

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

Important boundaries:

- `live-trade` does **not** read `config/project.yaml`
- `live-trade` creates `DataProcessor()` with the default `config/features_config.yaml`
- There is no CLI flag today to point live trading at a different features config path
- If `--risk-config` or `--position-manager-config` points to a missing file, live trading fails fast

## `config/model_config.yaml`

This file still matters because live trading and saved-model workflows use it directly.

### The Most Important Blocks

| Block | What it affects |
| --- | --- |
| `data` | Loader symbols, date ranges, cache behavior, primary timeframe |
| `news` | Optional news ingestion for sentiment pipelines |
| `training` | Train-time test split and CV folds |
| `trading` | Saved-model backtest and portfolio sizing defaults |

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

news:
  enabled: false
  provider: yfinance
  lookback_days: 30
  symbols: []

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

trading:
  initial_capital: 100000
  position_size: 0.2
  stop_loss: 0.02
  take_profit: 0.04
  max_positions: 5
  transaction_cost: 0.001
  max_risk_per_trade: 0.02
  max_portfolio_risk: 0.10
```

### Notes That Matter in Practice

- `news.provider` currently supports `yfinance`
- `training.cv_folds` is used for time-series CV during tuning
- `trading.initial_capital`, `max_risk_per_trade`, and `max_portfolio_risk` affect saved-model backtests
- `live-trade` takes `--initial-capital` and `--stop-loss-pct` as CLI overrides

## `config/features_config.yaml`

This file is still the source of truth for `DataProcessor`.

Example:

```yaml
price_features:
  - close_to_open
  - high_to_low
  - close_to_high
  - close_to_low
  - price_range

volume_features:
  - volume_sma:
      periods: [5, 10, 20]
  - volume_ema:
      periods: [5, 10, 20]
  - volume_sma_ratios: [5, 10, 20]
  - volume_ema_ratios: [5, 10, 20]
  - on_balance_volume
  - volume_price_trend

volatility_features:
  - atr_periods: [14]
  - bollinger_bands:
      period: 20
      std_dev: 2

custom_features:
  - mean_reversion:
      lookback: [5, 10, 20]

pipeline:
  steps:
    - generate_technical_indicators
    - generate_volume_features
    - generate_custom_features
    - handle_missing_values
    - remove_outliers
    - scale_features
    - select_features
```

### Useful Details

- `price_features` can be either a list or a mapping
- `volume_features`, `volatility_features`, and `custom_features` accept the list-of-mappings style above
- `generate_sentiment` only works when your data includes a `text` column
- `rolling_divergence` multi-timeframe operations require `rolling_window`
- During training, preprocessing is fit on the train split and reused on the test split

## `config/backtest_config.yaml`

This file drives execution modeling for `backtest` and `backtest-model`.

```yaml
data_path: data/backtest_sample.csv

execution:
  transaction_costs:
    enabled: true
    mode: bps
    value: 5
    apply_on: notional
  slippage:
    enabled: true
    mode: bps
    value: 10
    reference_price: close
  liquidity:
    enabled: false
    max_participation: 0.1
    volume_source: bar_volume
  impact:
    enabled: false
    model: linear
    alpha: 0.0
    beta: 0.0
    decay: 0.0
    spread: 0.0
    average_daily_volume: 0
  borrow_fee:
    enabled: false
    rate_bps: 0
  intrabar:
    enabled: false
    tick_column: ticks
    drift: 0.0
    volatility: 0.0
    synthetic_ticks: 0
```

### Override Order

For `backtest-model`, execution settings are resolved in this order:

1. Asset-class defaults from `config/impact_config.yaml`
2. Values from `config/backtest_config.yaml`
3. CLI overrides such as `--cost-bps` and `--slippage-fixed`

## `config/impact_config.yaml`

This file defines per-asset-class impact defaults used by saved-model backtests.

```yaml
asset_classes:
  equities:
    alpha: 0.1
    beta: 0.05
    model: linear
  futures:
    alpha: 0.2
    beta: 0.1
    model: square_root
```

Required keys per asset class:

- `alpha`
- `beta`
- `model`

## `config/risk_config.yaml`

This file configures drawdown protection and turnover limits.

```yaml
risk_management:
  drawdown_protection:
    enabled: true
    max_drawdown_pct: 0.15
    max_drawdown_absolute: 50000
    warning_threshold: 0.8
    soft_stop_threshold: 0.9
    hard_stop_threshold: 1.0
    emergency_stop_threshold: 1.1
    lookback_periods: [1, 7, 30]
  turnover_limits:
    daily_max: 2.0
    weekly_max: 5.0
    monthly_max: 15.0
```

Important behavior differences:

- `backtest-model` continues without a drawdown guard if `--risk-config` is missing or the file does not exist
- `live-trade` raises an error if the configured risk config path does not exist

## `config/streaming.yaml`

This file powers the built-in `StreamingGateway`.

```yaml
streaming:
  symbols: ["AAPL"]
  providers:
    - name: "alpaca"
      websocket_url: "wss://stream.data.alpaca.markets/v2/iex"
      auth_method: "api_key"
      subscriptions: ["trades", "quotes"]
      rate_limit:
        default_rate: 100
        burst_allowance: 50
      circuit_breaker:
        failure_threshold: 5
        timeout: 30
  buffer_size: 10000
  reconnect_attempts: 5
  health_check_interval: 30

streaming_health:
  monitoring:
    check_interval: 5
  thresholds:
    max_latency_ms: 100
    min_throughput_msg_per_sec: 50
    max_reconnect_attempts: 5
    max_queue_depth: 5000
    circuit_breaker_timeout: 60
  alerts:
    channels: ["log", "metrics"]
    escalation_threshold: 3
  metrics:
    enabled: true
    host: "0.0.0.0"
    port: 9000
  api:
    enabled: false
    host: "0.0.0.0"
    port: 8000
```

### `streaming` Keys That Matter Today

| Key | What it does |
| --- | --- |
| `symbols` | Default subscription symbols |
| `providers[].name` | Built-in adapter name. Current gateway support: `alpaca`, `interactive_brokers` |
| `providers[].websocket_url` | Provider endpoint |
| `providers[].auth_method` | Auth method passed to the websocket manager |
| `providers[].subscriptions` | Channels to subscribe to, typically `trades` and `quotes` |
| `providers[].symbols` | Optional per-provider override for the symbol list |
| `providers[].rate_limit` | Rate-limit config passed to the websocket manager |
| `providers[].circuit_breaker` | Circuit-breaker config passed to the websocket manager |
| `buffer_size` | Stream buffer queue size |
| `reconnect_attempts` | Connection retry budget |
| `health_check_interval` | Optional connection-pool health loop interval |

### `streaming_health` Keys That Matter Today

| Key | What it does |
| --- | --- |
| `monitoring.check_interval` | Streaming health poll interval |
| `thresholds.max_latency_ms` | Latency warning threshold |
| `thresholds.min_throughput_msg_per_sec` | Throughput warning threshold |
| `thresholds.max_reconnect_attempts` | Recovery manager reconnect threshold |
| `thresholds.max_queue_depth` | Queue depth warning threshold |
| `thresholds.circuit_breaker_timeout` | Recovery reset timeout |
| `alerts.channels` | Alert sinks |
| `alerts.escalation_threshold` | Escalation count before promoting warnings |
| `metrics.enabled` | Enables the Prometheus exporter |
| `metrics.host` / `metrics.port` | Prometheus bind address |
| `api.enabled` | Enables the health API |
| `api.host` / `api.port` | Health API bind address |

Important current limitation:

- `streaming_health.monitoring.enabled` and `streaming_health.monitoring.metrics_retention` appear in sample configs but are not currently consulted by the gateway
- `streaming_health.alerts.enabled` is also not currently used by the gateway

### Health API and Metrics Interaction

- If `streaming_health.metrics.enabled` is `true`, the gateway starts a Prometheus exporter
- If the metrics exporter and the health API are configured on the same host and port, the dedicated exporter is skipped to avoid double-binding
- The health API still serves `/health`, `/status`, and `/metrics` when enabled

## `config/position_manager.yaml`

This file configures the thread-safe `PositionManager` used by the live-trading path.

```yaml
position_manager:
  risk_management:
    drawdown_protection:
      enabled: true
      max_drawdown_pct: 0.2
  impact:
    enabled: true
    model: linear
    alpha: 0.1
    beta: 0.05
  reconciliation:
    intraday: "1m"
    daily: "1d"
  mode: paper
```

### What Each Block Does

| Block | What it does today |
| --- | --- |
| `risk_management` | Enables drawdown and turnover rules inside the position manager |
| `impact` | Builds a market-impact calculator for position-manager analytics |
| `reconciliation` | Labels the reconciliation windows used by `reconcile_positions()` |
| `mode` | Marks the environment as `paper` or `live` |

Important current limitation:

- The built-in `live-trade` flow opens and closes positions without passing ADV values into the position manager
- That means the `impact` block is accepted and instantiated, but its richer cost modeling is not fully exercised by the default live path

## Validation Commands

Validate the runtime YAML bundle:

```bash
poetry run quanttradeai validate-config --output-dir reports/config_validation
```

This writes a consolidated JSON and CSV summary and exits non-zero on failure.

## Bottom Line

If you are working on research or agent design, stay in `project.yaml`.

If you are running live or saved-model workflows today, keep these runtime YAMLs healthy and validate them before you run.
