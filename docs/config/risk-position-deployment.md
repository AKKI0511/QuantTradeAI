# Risk, Position Manager, And Deployment

`risk`, `position_manager`, and `deployment` group the settings that matter when moving from experiments toward paper or live operation. They do not make a deployment trade by themselves; they configure runtime guardrails and bundle generation.

## Used By

| Command | Fields used |
|---|---|
| `quanttradeai validate` | Enforces live prerequisites when any agent is configured with `mode: live`. |
| `quanttradeai agent run --mode live` | Compiles `runtime_risk_config.yaml` and `runtime_position_manager_config.yaml`. |
| `quanttradeai promote --to live` | Requires streaming, `risk`, `position_manager`, and `--acknowledge-live <agent>`. |
| `quanttradeai deploy --agent <name>` | Reads `deployment.target` and `deployment.mode`, then validates mode-specific runtime requirements. |

## Supported Fields

### `risk`

```yaml
risk:
  drawdown_protection:
    enabled: true
    max_drawdown_pct: 0.15
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

| Field | Notes |
|---|---|
| `drawdown_protection.enabled` | Must be `true` when an agent is configured `mode: live`. |
| `drawdown_protection.max_drawdown_pct` | Percentage drawdown limit. |
| `drawdown_protection.max_drawdown_absolute` | Absolute drawdown limit. |
| `warning_threshold` | Ratio that marks warning state. Defaults to `0.8`. |
| `soft_stop_threshold` | Ratio that reduces position size. Defaults to `0.9`. |
| `hard_stop_threshold` | Ratio that halts trading. Defaults to `1.0`. |
| `emergency_stop_threshold` | Ratio that marks emergency stop. Defaults to `1.1`. |
| `lookback_periods` | Rolling max drawdown windows. Defaults to `[1, 7, 30]`. |
| `turnover_limits.daily_max` | Daily turnover cap. |
| `turnover_limits.weekly_max` | Weekly turnover cap. |
| `turnover_limits.monthly_max` | Monthly turnover cap. |

### `position_manager`

```yaml
position_manager:
  impact:
    enabled: false
    model: linear
    alpha: 0.0
    beta: 0.0
  reconciliation:
    intraday: 1m
    daily: 1d
  mode: live
```

| Field | Supported values |
|---|---|
| `impact.enabled` | Boolean. |
| `impact.model` | `linear`, `square_root`, or `almgren_chriss`. |
| `impact.alpha`, `impact.beta`, `impact.gamma` | Impact model parameters. |
| `impact.decay`, `impact.spread` | Execution cost controls. |
| `reconciliation` | Mapping of labels to timeframes. Defaults to `intraday: 1m`, `daily: 1d`. |
| `mode` | `paper` or `live`; project templates use `live` for live runtime compilation. |

Live runtime compilation injects the top-level `risk` section into the generated position manager runtime config.

### `deployment`

```yaml
deployment:
  target: docker-compose
  mode: paper
```

| Field | Supported values | Notes |
|---|---|---|
| `deployment.target` | `docker-compose`, `local`, `render` | Used by `quanttradeai deploy` unless `--target` overrides it. |
| `deployment.mode` | `paper`, `live` | Used by `quanttradeai deploy` unless `--mode` overrides it. |

## Deployment Targets

| Target | Output |
|---|---|
| `docker-compose` | `docker-compose.yml`, `Dockerfile`, `.env.example`, `README.md`, `resolved_project_config.yaml`, `deployment_manifest.json`. |
| `local` | `run.py`, `.env.example`, `README.md`, `resolved_project_config.yaml`, `deployment_manifest.json`. |
| `render` | `render.yaml`, `Dockerfile`, `.env.example`, `README.md`, `assets/`, `resolved_project_config.yaml`, `deployment_manifest.json`. |

Default output path:

```text
reports/deployments/<agent>/<timestamp>/
```

Use `--output <dir>` to choose a bundle directory. Existing non-empty output directories require `--force`.

## Paper And Live Boundaries

| Boundary | Behavior |
|---|---|
| Paper agent run | Can be replay-backed or real-time. Execution is simulated unless `execution.backend: alpaca` is configured. |
| Paper deployment bundle | Always uses real-time streaming; replay is disabled in the generated resolved config. |
| Live agent run | Requires real-time streaming, agent `mode: live`, top-level `risk`, and top-level `position_manager`. |
| Live deployment bundle | Requires the selected agent to already be configured `mode: live`. |
| Broker-backed execution | Controlled by `agents[].execution.backend: alpaca`, not by `deployment.mode` alone. |

Broker-backed Alpaca execution requires:

- `agents[].execution.backend: alpaca`
- `data.streaming.provider: alpaca`
- `data.streaming.enabled: true`
- real-time paper or live streaming
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`

## Examples

### Live-Ready Agent Sections

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
    symbols: [AAPL]
    channels: [trades, quotes]
    replay:
      enabled: false

agents:
  - name: rsi_reversion
    kind: rule
    mode: live
    execution:
      backend: simulated
    rule:
      preset: rsi_threshold
      feature: rsi_14
      buy_below: 30
      sell_above: 70
    context:
      features: [rsi_14]

risk:
  drawdown_protection:
    enabled: true
    max_drawdown_pct: 0.15
  turnover_limits:
    daily_max: 2.0
    weekly_max: 5.0
    monthly_max: 15.0

position_manager:
  impact:
    enabled: false
    model: linear
  reconciliation:
    intraday: 1m
    daily: 1d
  mode: live

deployment:
  target: docker-compose
  mode: live
```

### Deployment Commands

```bash
quanttradeai deploy --agent rsi_reversion -c config/project.yaml
quanttradeai deploy --agent rsi_reversion -c config/project.yaml --target local --mode paper
quanttradeai deploy --agent rsi_reversion -c config/project.yaml --target render --mode live
```

## Expected Outcomes

Live agent runs write:

- `runtime_risk_config.yaml`
- `runtime_position_manager_config.yaml`
- `runtime_streaming_config.yaml`
- `decisions.jsonl`
- `executions.jsonl`
- `metrics.json`
- `summary.json`

Deployment bundles write:

- `resolved_project_config.yaml`
- `deployment_manifest.json`
- target-specific files such as `docker-compose.yml`, `run.py`, or `render.yaml`
- `.env.example` with inferred provider environment variables
- copied `assets/` for Render when prompts, notes, or model artifacts are needed

## Common Mistakes

| Mistake | Result |
|---|---|
| Treating deployment bundle generation as live trading | `deploy` only writes files; trading starts when the generated command is run. |
| Missing `risk` or `position_manager` for live | Validation, live runs, or live deployment fails. |
| Promoting to live without `--acknowledge-live <agent>` | Promotion fails. |
| Thinking paper success approves live automatically | Paper success is only an artifact; live still requires explicit promotion and live config. |
| Generating a paper deployment from replay-only streaming settings | Deployment fails because paper bundles require real-time provider fields. |
| Assuming `deployment.mode: live` enables broker execution | Broker execution requires `agents[].execution.backend: alpaca`. |
