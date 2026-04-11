# Project Config (`project.yaml`)

`config/project.yaml` is the canonical config entrypoint for QuantTradeAI.

It drives:

- `quanttradeai init`
- `quanttradeai validate`
- `quanttradeai research run`
- `quanttradeai agent run` for project-defined agents in `backtest`, `paper`, and `live`
- `quanttradeai promote` for research-model promotion plus agent backtest-to-paper and paper-to-live promotion
- `quanttradeai deploy` for docker-compose paper-agent bundles

`live-trade` still uses the legacy runtime YAML files documented in [Runtime and Live Trading Configs](live-runtime-files.md).

## Supported Project Workflows

### Research

```bash
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai promote --run research/<run_id> -c config/project.yaml
```

### Model Agents

```bash
poetry run quanttradeai init --template model-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode backtest
poetry run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode paper
poetry run quanttradeai promote --run agent/paper/<run_id> -c config/project.yaml --to live --acknowledge-live paper_momentum
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode live
```

The `model-agent` template also creates a placeholder model artifact at `models/promoted/aapl_daily_classifier/README.md`. Replace that directory with a promoted research model artifact or another compatible saved model before you run the agent.

### Rule Agents

```bash
poetry run quanttradeai init --template rule-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent rsi_reversion -c config/project.yaml --mode backtest
poetry run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
poetry run quanttradeai agent run --agent rsi_reversion -c config/project.yaml --mode paper
poetry run quanttradeai promote --run agent/paper/<run_id> -c config/project.yaml --to live --acknowledge-live rsi_reversion
poetry run quanttradeai agent run --agent rsi_reversion -c config/project.yaml --mode live
```

### LLM And Hybrid Agents

```bash
poetry run quanttradeai init --template llm-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
poetry run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode paper
poetry run quanttradeai promote --run agent/paper/<run_id> -c config/project.yaml --to live --acknowledge-live breakout_gpt
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode live
```

The `llm-agent` and `hybrid` templates also create starter prompt files under `prompts/` so validation can pass immediately.

Hybrid projects add the research promotion handoff before agent runs:

```bash
poetry run quanttradeai init --template hybrid -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai promote --run research/<run_id> -c config/project.yaml
poetry run quanttradeai agent run --agent hybrid_swing_agent -c config/project.yaml --mode backtest
poetry run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
poetry run quanttradeai agent run --agent hybrid_swing_agent -c config/project.yaml --mode paper
poetry run quanttradeai promote --run agent/paper/<run_id> -c config/project.yaml --to live --acknowledge-live hybrid_swing_agent
poetry run quanttradeai agent run --agent hybrid_swing_agent -c config/project.yaml --mode live
```

The default hybrid template already points `model_signal_sources` at `models/promoted/aapl_daily_classifier`, so the happy path does not require editing timestamped experiment directories.

Current support:

| Agent kind | Backtest | Paper | Live |
| --- | --- | --- | --- |
| `model` | Yes | Yes | Yes |
| `llm` | Yes | Yes | Yes |
| `hybrid` | Yes | Yes | Yes |
| `rule` | Yes | Yes | Yes |

Successful research runs can promote trained model artifacts into stable project paths with:

```bash
poetry run quanttradeai promote --run research/<run_id> -c config/project.yaml
```

Research promotion copies the configured symbol artifact directories from the run's `artifacts.experiment_dir` into each `research.promotion.targets[].path` and writes a `promotion_manifest.json` file in each promoted destination.

Successful agent backtest runs can be promoted to paper mode with:

```bash
poetry run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
```

Successful agent paper runs can be promoted to live mode with:

```bash
poetry run quanttradeai promote --run agent/paper/<run_id> -c config/project.yaml --to live --acknowledge-live <agent_name>
```

Research promotion updates model directories only and does not change `deployment.mode`.
Paper promotion updates the matching agent's `mode` and `deployment.mode` to `paper`.
Live promotion updates only the matching agent's `mode` to `live`. `deployment.mode` stays unchanged because live deployment is still out of scope for the canonical workflow.

Docker Compose deployment bundles can be generated with:

```bash
poetry run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target docker-compose
```

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
  promotion:
    targets:
      - name: "aapl_daily_classifier"
        symbol: "AAPL"
        path: "models/promoted/aapl_daily_classifier"

agents:
  - name: "paper_momentum"
    kind: "model"
    mode: "paper"
    model:
      path: "models/promoted/aapl_daily_classifier"
    context:
      features: ["rsi_14"]
      positions: true
      risk_state: true
    risk:
      max_position_pct: 0.05

deployment:
  target: "docker-compose"
  mode: "paper"

risk:
  drawdown_protection:
    enabled: true
    max_drawdown_pct: 0.1
    warning_threshold: 0.8
    soft_stop_threshold: 0.9
    hard_stop_threshold: 1.0
    emergency_stop_threshold: 1.05
  turnover_limits:
    daily_max: 2.0
    weekly_max: 5.0
    monthly_max: 12.0

position_manager:
  impact:
    enabled: false
    model: "linear"
    alpha: 0.0
    beta: 0.0
  reconciliation:
    intraday: "1m"
    daily: "1d"
  mode: "live"
```

Rule-agent example:

```yaml
agents:
  - name: "rsi_reversion"
    kind: "rule"
    mode: "paper"
    rule:
      preset: "rsi_threshold"
      feature: "rsi_14"
      buy_below: 30.0
      sell_above: 70.0
    context:
      features: ["rsi_14"]
      positions: true
      risk_state: true
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

Used by project-defined paper and live agents.

Required when any agent is configured with `mode: paper` or `mode: live`:

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

Validation enforces the same rule for both paper and live agents: if any agent is configured with `mode: paper` or `mode: live`, then `data.streaming.enabled` must be `true`.

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
- `promotion.targets`

Practical notes:

- `evaluation.use_configured_test_window: true` keeps `data.test_start` and `data.test_end` in the compiled runtime model config
- setting it to `false` clears those fields and lets the downstream research runtime fall back to its chronological split behavior
- `model.tuning.enabled` and `model.tuning.trials` flow directly into the training run
- `backtest.costs.bps` becomes the transaction cost setting in the compiled runtime backtest config
- `promotion.targets` maps promoted symbol artifacts into stable project-relative destinations under `models/`
- each promotion target requires `name`, `symbol`, and `path`
- `promotion.targets[].symbol` must exist in `data.symbols`
- `promotion.targets[].path` must be project-relative and resolve under `models/`
- promotion target names and paths must be unique

Even when `research.enabled` is `false`, the agent runtime still reuses these defaults to compile consistent runtime configs.

### `agents`

#### `rule` agents

Required fields:

- `name`
- `kind: rule`
- `rule.preset`
- `rule.feature`
- `rule.buy_below`
- `rule.sell_above`

Current built-in preset support:

- `rsi_threshold`

Validation checks that:

- the `rule` block exists for rule agents
- `rule.feature` exists in `features.definitions`
- `rule.feature` is also listed in `context.features`
- `rule.buy_below < rule.sell_above`

Backtest mode:

- evaluates the configured feature on each completed bar
- writes `summary.json`, `metrics.json`, `equity_curve.csv`, `decisions.jsonl`, and `ledger.csv` when trades are present
- also writes compiled runtime config files and per-symbol coverage and metrics files under the run directory
- does not emit prompt sample artifacts because no LLM is involved

Paper mode:

- compiles `data.streaming` into a runtime streaming config
- warm-starts with recent historical OHLCV before streaming begins
- evaluates the configured rule on each completed streaming bar
- writes `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, and `runtime_streaming_config.yaml`
- does not emit `prompt_samples.json`

Live mode:

- requires the agent itself to already be configured with `mode: live`
- rejects `--skip-validation`
- compiles `data.streaming`, top-level `risk`, and top-level `position_manager` into runtime YAML snapshots inside the run directory
- writes `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, `runtime_streaming_config.yaml`, `runtime_risk_config.yaml`, and `runtime_position_manager_config.yaml`

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

Live mode:

- requires `mode: live` in the agent config before `quanttradeai agent run --mode live` is allowed
- rejects `--skip-validation`
- compiles runtime streaming, risk, and position-manager YAML snapshots from `project.yaml`
- writes `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, `runtime_streaming_config.yaml`, `runtime_risk_config.yaml`, and `runtime_position_manager_config.yaml`

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

Live mode:

- requires `mode: live` in the agent config before `quanttradeai agent run --mode live` is allowed
- rejects `--skip-validation`
- compiles runtime streaming, risk, and position-manager YAML snapshots from `project.yaml`
- writes `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, `prompt_samples.json`, `runtime_streaming_config.yaml`, `runtime_risk_config.yaml`, and `runtime_position_manager_config.yaml`

### `deployment`

Deployment metadata for the canonical project workflow.

Current happy-path support:

- `target: "docker-compose"`
- `mode: "paper"`

Example:

```yaml
deployment:
  target: "docker-compose"
  mode: "paper"
```

Behavior:

- `quanttradeai promote --run research/<run_id> -c config/project.yaml` copies trained model artifacts into stable `models/...` destinations and writes `promotion_manifest.json`
- `quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml` updates `deployment.mode` to `paper` when promoting a successful agent backtest run
- `quanttradeai deploy --agent <name> -c config/project.yaml --target docker-compose` generates a bundle under `reports/deployments/<agent>/<timestamp>/`
- generated bundles include `docker-compose.yml`, `Dockerfile`, `.env.example`, `README.md`, `resolved_project_config.yaml`, and `deployment_manifest.json`
- generated compose services run `quanttradeai agent run --agent <name> -c config/project.yaml --mode paper`

Live deployment remains future work.

### `risk`

Top-level live guardrails used by project-defined live agents.

Canonical live usage:

- `risk.drawdown_protection.enabled` must be `true`
- `drawdown_protection` drives drawdown safety states
- `turnover_limits` drives trade-throttling thresholds

Compatibility note:

- `position_manager.risk_management` is accepted as a legacy compatibility input during validation
- the top-level `risk` block is the canonical source going forward

### `position_manager`

Top-level runtime settings for project-defined live execution.

Used for:

- execution impact settings
- reconciliation cadence
- runtime mode

The canonical live compiler injects the top-level `risk` block into the generated live runtime position-manager YAML.

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

Successful research runs also record `artifacts.experiment_dir` in `summary.json`, which is the source used by `quanttradeai promote --run research/<run_id>`.

### `quanttradeai promote`

Research promotion does not create a new run directory. It copies the configured symbol model directories into stable destinations under `models/` and writes `promotion_manifest.json` in each promoted destination.

Agent promotion updates `config/project.yaml` in place. Backtest-to-paper promotion also updates `deployment.mode` to `paper`; paper-to-live promotion changes only the matching agent's mode.

### `quanttradeai agent run --mode backtest`

Writes under `runs/agent/backtest/<timestamp>_<agent>/`.

### `quanttradeai agent run --mode paper`

Writes under `runs/agent/paper/<timestamp>_<agent>/`.

For `llm` and `hybrid` paper runs, the run directory includes both `decisions.jsonl` and `executions.jsonl` so prompt-driven decisions and actual paper executions can be audited separately.

### `quanttradeai agent run --mode live`

Writes under `runs/agent/live/<timestamp>_<agent>/`.

Live run directories include:

- `resolved_project_config.yaml`
- `summary.json`
- `metrics.json`
- `decisions.jsonl`
- `executions.jsonl`
- `runtime_streaming_config.yaml`
- `runtime_risk_config.yaml`
- `runtime_position_manager_config.yaml`
- `prompt_samples.json` for `llm` and `hybrid`

## Compatibility Notes

The canonical compiler still accepts some compatibility sections while the repo transitions from legacy YAMLs:

- `news`
- `training`
- `trading`
- `execution`
- `models`

Treat those as migration aids, not the preferred long-term shape.
