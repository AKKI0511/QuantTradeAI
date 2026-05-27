# Project File

The project YAML file is the canonical workspace definition. The default generated path is `config/project.yaml`, but the file does not need to be named `project.yaml`; commands that accept project config also accept `-c <path>`.

```bash
quanttradeai validate -c config/momentum.yaml
quanttradeai agent run --agent rsi_reversion -c config/aapl-paper.yaml --mode paper
```

## Used By

| Command | How it uses the file |
|---|---|
| `quanttradeai validate -c <path>` | Validates the project file and writes a resolved config snapshot. |
| `quanttradeai research run -c <path>` | Compiles research runtime configs, trains models, runs model backtests, and writes run artifacts. |
| `quanttradeai agent run --agent <name> -c <path>` | Loads one YAML-defined trading agent and runs it in `backtest`, `paper`, or `live` mode. |
| `quanttradeai agent run --all -c <path>` | Runs every configured agent as a batch. |
| `quanttradeai agent run --sweep <name> -c <path>` | Expands an `agent_backtest` sweep and runs the child backtests. |
| `quanttradeai promote --run <id> -c <path>` | Updates project YAML for agent mode promotion or copies trained research models to promotion targets. |
| `quanttradeai deploy --agent <name> -c <path>` | Generates a local, Docker Compose, or Render deployment bundle for an agent. |

## Supported Fields

The validator requires these top-level sections:

| Section | Required | Purpose |
|---|---:|---|
| `project` | Yes | Project name and active profile label. |
| `profiles` | Yes | Named profile metadata. |
| `data` | Yes | Historical data window, cache behavior, and streaming settings. |
| `features` | Yes | Feature definitions compiled into runtime feature config. |
| `research` | Yes | Research labels, model settings, evaluation, backtest costs, and promotion targets. |
| `agents` | Yes | YAML-defined trading agents. |
| `deployment` | Yes | Default deployment target and mode. |
| `sweeps` | No | Research or agent backtest parameter grids. |
| `risk` | Required for live | Live drawdown and turnover controls. |
| `position_manager` | Required for live | Live position tracking and impact settings. |
| `news` | Only when used | Enables news-backed LLM/hybrid context and optional news ingestion. |

Path-like fields such as prompts, notes, model artifacts, and promotion targets are resolved relative to the inferred project root. If the config file is inside `config/`, the project root is the parent of `config/`; otherwise it is the config file's parent directory.

## Skeleton

```yaml
project:
  name: strategy_lab
  profile: paper

profiles:
  research: {mode: research}
  paper: {mode: paper}
  live: {mode: live}

data:
  symbols: [AAPL]
  start_date: "2022-01-01"
  end_date: "2024-12-31"
  timeframe: 1d
  test_start: "2024-09-01"
  test_end: "2024-12-31"
  use_cache: true
  refresh: false
  max_workers: 1
  streaming:
    enabled: true
    provider: alpaca
    websocket_url: wss://stream.data.alpaca.markets/v2/iex
    auth_method: api_key
    symbols: [AAPL]
    channels: [trades, quotes]
    replay:
      enabled: true
      pace_delay_ms: 0

features:
  definitions:
    - name: rsi_14
      type: technical
      params:
        period: 14
    - name: sma_20
      type: technical
      params: {}

research:
  enabled: true
  labels:
    type: forward_return
    horizon: 5
    buy_threshold: 0.01
    sell_threshold: -0.01
  model:
    kind: classifier
    family: voting
    tuning:
      enabled: true
      trials: 50
  evaluation:
    split: time_aware
    use_configured_test_window: true
  backtest:
    costs:
      enabled: true
      bps: 5
  promotion:
    targets:
      - name: aapl_daily_classifier
        symbol: AAPL
        path: models/promoted/aapl_daily_classifier

agents:
  - name: rsi_reversion
    kind: rule
    mode: paper
    execution:
      backend: simulated
    rule:
      preset: rsi_threshold
      feature: rsi_14
      buy_below: 30
      sell_above: 70
    context:
      features: [rsi_14]
      positions: true
      risk_state: true
    tools: []
    risk:
      max_position_pct: 0.05

sweeps:
  - name: rsi_threshold_grid
    kind: agent_backtest
    agent: rsi_reversion
    parameters:
      - path: rule.buy_below
        values: [25, 30]

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
  mode: paper
```

## Expected Outcomes

`validate` writes:

- `reports/config_validation/<timestamp>/resolved_project_config.yaml`
- `reports/config_validation/<timestamp>/summary.json`

Research and agent runs write run directories under `runs/` with:

- `summary.json`
- `metrics.json`
- runtime config snapshots such as `runtime_model_config.yaml`, `runtime_features_config.yaml`, and `runtime_streaming_config.yaml`
- mode-specific artifacts such as decisions, executions, equity curves, replay manifests, and scoreboards

Deployment writes bundle directories under `reports/deployments/<agent>/<timestamp>/` unless `--output` is provided.

## Common Mistakes

| Mistake | Result |
|---|---|
| Omitting a required top-level section | `validate` fails before any run starts. |
| Assuming the file must be named `project.yaml` | Any YAML path can be used with `-c`, but relative asset paths resolve from that config's project root. |
| Editing checked-in YAML during a sweep | Sweeps create in-memory and child-run variants; the source YAML is not mutated. |
| Configuring `mode: live` without `risk` and `position_manager` | Validation and live deployment fail. |
| Using an agent name that is not in `agents` | `agent run`, `deploy`, and promotion fail with an agent lookup error. |
