# Data And Streaming

The `data` section controls historical bars, test windows, caching, and real-time or replay streaming. Research runs use the historical fields. Paper/live agent runs and deployment bundles use the streaming fields.

## Used By

| Command | Fields used |
|---|---|
| `quanttradeai validate` | Validates date windows, streaming requirements, replay windows, and symbol shape. |
| `quanttradeai research run` | Compiles `data` into `runtime_model_config.yaml` for data loading and time-aware splits. |
| `quanttradeai agent run --mode backtest` | Loads historical data and uses `test_start`/`test_end` for the backtest window. |
| `quanttradeai agent run --mode paper` | Requires `data.streaming.enabled`; uses replay if configured, otherwise real-time streaming. |
| `quanttradeai agent run --mode live` | Requires real-time streaming provider, websocket URL, channels, risk, and position manager. |
| `quanttradeai deploy` | Requires real-time streaming fields for paper and live bundles. |

## Supported Fields

### Historical Data

| Field | Type | Notes |
|---|---|---|
| `data.symbols` | list | Strings such as `AAPL`, or mappings with `ticker` and optional `asset_class`. |
| `data.start_date` | string | ISO date. Required. |
| `data.end_date` | string | ISO date. Required and must be on or after `start_date`. |
| `data.timeframe` | string | Defaults to `1d` when omitted in runtime compilation. |
| `data.test_start` | string | Optional ISO date inside the data window. |
| `data.test_end` | string | Optional ISO date inside the data window and on or after `test_start`. |
| `data.cache_path` | string | Runtime cache directory. Defaults to `data/raw`. |
| `data.cache_dir` | string | Accepted cache directory field; `cache_path` takes precedence at load time. |
| `data.cache_expiration_days` | integer | Cache age limit. `null` means an existing cache file does not expire. |
| `data.use_cache` | boolean | Defaults to `true`. |
| `data.refresh` | boolean | Defaults to `false`; when true, data loading bypasses cached bars. |
| `data.max_workers` | integer | Defaults to `1`; values above 1 fetch symbols concurrently. |

`data.symbols` can include asset classes:

```yaml
data:
  symbols:
    - ticker: AAPL
      asset_class: equities
    - MSFT
  start_date: "2021-01-01"
  end_date: "2024-12-31"
  timeframe: 1d
```

### Streaming

| Field | Type | Notes |
|---|---|---|
| `data.streaming.enabled` | boolean | Required for configured paper/live agents. |
| `data.streaming.provider` | string | Required for real-time paper, live, and deployment bundles. |
| `data.streaming.websocket_url` | string | Required for real-time paper, live, and deployment bundles. |
| `data.streaming.auth_method` | string | Defaults to `api_key`. |
| `data.streaming.symbols` | list | Overrides stream symbols; falls back to `data.symbols`. |
| `data.streaming.channels` | list | Supported values are `trades` and `quotes`; required when streaming is enabled. |
| `data.streaming.buffer_size` | integer | Defaults to `1000`. |
| `data.streaming.reconnect_attempts` | integer | Defaults to `5`. |
| `data.streaming.health_check_interval` | integer | Included in runtime streaming config when present. |
| `data.streaming.rate_limit` | mapping | Passed through to the runtime provider config when present. |
| `data.streaming.circuit_breaker` | mapping | Passed through to the runtime provider config when present. |

Health and monitoring sections are passed into `streaming_health` in the runtime config:

| Section | Supported fields |
|---|---|
| `monitoring` | `enabled`, `check_interval`, `metrics_retention` |
| `thresholds` | `max_latency_ms`, `min_throughput_msg_per_sec`, `max_reconnect_attempts`, `max_queue_depth`, `circuit_breaker_timeout` |
| `alerts` | `enabled`, `channels`, `escalation_threshold` |
| `metrics` | `enabled`, `host`, `port` |
| `api` | `enabled`, `host`, `port` |

### Replay

| Field | Type | Notes |
|---|---|---|
| `data.streaming.replay.enabled` | boolean | Enables replay-backed paper mode. |
| `data.streaming.replay.start_date` | string | Optional ISO date. |
| `data.streaming.replay.end_date` | string | Optional ISO date. |
| `data.streaming.replay.pace_delay_ms` | integer | Delay between replayed bars. Defaults to `0`. |

Replay date resolution is deterministic:

1. Use explicit `data.streaming.replay.start_date` and `end_date` when provided.
2. Otherwise use `data.test_start` and `data.test_end`.
3. Otherwise use `data.start_date` and `data.end_date`.

The resolved replay window must stay inside the full data window.

## Examples

### Research/Backtest Historical Data

```yaml
data:
  symbols: [AAPL, MSFT]
  start_date: "2019-01-01"
  end_date: "2024-12-31"
  timeframe: 1d
  test_start: "2024-09-01"
  test_end: "2024-12-31"
  cache_path: data/raw
  cache_expiration_days: 7
  use_cache: true
  refresh: false
```

### Replay-Backed Paper Mode

```yaml
data:
  symbols: [AAPL]
  start_date: "2024-01-01"
  end_date: "2024-03-31"
  test_start: "2024-03-01"
  test_end: "2024-03-15"
  timeframe: 1d
  streaming:
    enabled: true
    symbols: [AAPL]
    channels: [trades, quotes]
    replay:
      enabled: true
      pace_delay_ms: 0
```

With replay enabled, paper runs can use historical bars without `provider` or `websocket_url`. Deployment bundles are different: paper bundles always require real-time fields because replay is disabled in the generated bundle.

### Real-Time Paper Or Live Streaming

```yaml
data:
  symbols: [AAPL]
  start_date: "2022-01-01"
  end_date: "2024-12-31"
  timeframe: 1d
  streaming:
    enabled: true
    provider: alpaca
    websocket_url: wss://stream.data.alpaca.markets/v2/iex
    auth_method: api_key
    symbols: [AAPL]
    channels: [trades, quotes]
    buffer_size: 1000
    reconnect_attempts: 5
    replay:
      enabled: false
```

## Expected Outcomes

Research and backtest runs write:

- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `runtime_backtest_config.yaml`
- per-run data validation output unless `--skip-validation` is used

Paper/live runs write:

- `runtime_streaming_config.yaml`
- `replay_manifest.json` when replay-backed paper mode is used
- `decisions.jsonl`, `executions.jsonl`, `metrics.json`, and `summary.json`

Deployment bundles write a resolved project config snapshot. Paper deployment bundles disable replay in that snapshot and require real-time streaming fields.

## Common Mistakes

| Mistake | Result |
|---|---|
| Setting replay dates outside `data.start_date` and `data.end_date` | Validation fails. |
| Enabling live mode without `provider`, `websocket_url`, or `channels` | Validation or runtime compilation fails. |
| Assuming `paper` always means broker-backed paper | Paper mode is simulated unless `agents[].execution.backend: alpaca` is configured. |
| Combining replay-backed paper mode with `execution.backend: alpaca` | Validation fails; Alpaca-backed paper requires real-time streaming. |
| Generating a paper deployment from a replay-only config | Deployment fails because bundles require real-time streaming fields. |

