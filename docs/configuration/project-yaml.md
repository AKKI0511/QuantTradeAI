# Project Config (`project.yaml`)

`config/project.yaml` is the canonical config entrypoint for QuantTradeAI.

It drives:

- `quanttradeai init`
- `quanttradeai validate`
- `quanttradeai research run`
- `quanttradeai agent run` for project-defined agents in `backtest`, `paper`, and `live`
- `quanttradeai agent run --all` for multi-agent batches in `backtest`, `paper`, and `live`
- `quanttradeai agent run --sweep <name>` for backtest-only parameter sweeps
- `quanttradeai promote` for research-model promotion, sweep winner materialization, and agent backtest-to-paper/paper-to-live promotion
- `quanttradeai deploy` for local, docker-compose, and Render paper/live agent bundles

For local project-defined agents, `agent run --mode paper` defaults to deterministic replay when `data.streaming.replay.enabled: true`.

If an agent sets `execution.backend: alpaca`, QuantTradeAI switches paper/live execution from local simulated fills to Alpaca-backed market orders. That happy path requires `data.streaming.enabled: true`, `data.streaming.provider: alpaca`, and a real-time paper/live run instead of replay-backed paper mode.

## Supported Project Workflows

### Research

```bash
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai promote --run research/<run_id> -c config/project.yaml
```

### Strategy Lab

```bash
poetry run quanttradeai init --template strategy-lab -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --all -c config/project.yaml --mode backtest --max-concurrency 4
poetry run quanttradeai agent run --sweep rsi_threshold_grid -c config/project.yaml --mode backtest --max-concurrency 4
poetry run quanttradeai agent run --sweep sma_risk_grid -c config/project.yaml --mode backtest --max-concurrency 4
poetry run quanttradeai runs list --scoreboard --sort-by net_sharpe
poetry run quanttradeai promote --run agent/backtest/<winning_run_id> -c config/project.yaml
```

The `strategy-lab` template defines two deterministic rule agents, `rsi_reversion` and `sma_trend`, plus two starter sweeps. It is the quickest YAML-only path for comparing multiple strategies before promoting a winner.

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

### Multi-Agent Batches

```bash
poetry run quanttradeai agent run --all -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --all -c config/project.yaml --mode paper
poetry run quanttradeai agent run --all -c config/project.yaml --mode paper --max-concurrency 4
poetry run quanttradeai agent run --all -c config/project.yaml --mode live --acknowledge-live <project_name>
```

Multi-agent batches preserve the normal child runs under `runs/agent/backtest/...`, `runs/agent/paper/...`, or `runs/agent/live/...` and write batch artifacts under `runs/agent/batches/<timestamp>_<project>_<mode>/`.

Every batch directory includes `summary.json`, `experiment_brief.json`, and `experiment_brief.md`. The JSON brief identifies the top successful run from the existing scoreboard order, failed children, useful logs, and exact next commands for promotion or inspection. Batch summaries are discoverable with `quanttradeai runs list --type batch --json`.

Backtest batches rank by `net_sharpe`. Paper and live batches rank by `total_pnl`. Live batches require every configured agent to already have `mode: live` and require `--acknowledge-live` to match `project.name`.

### Backtest Sweeps

```bash
poetry run quanttradeai agent run --sweep rsi_threshold_grid -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --sweep rsi_threshold_grid -c config/project.yaml --mode backtest --max-concurrency 4
poetry run quanttradeai runs list --scoreboard --sort-by net_sharpe
poetry run quanttradeai promote --run agent/backtest/<winner_run_id> -c config/project.yaml
poetry run quanttradeai agent run --agent rsi_reversion -c config/project.yaml --mode paper
```

Sweeps expand one project-defined agent into many backtest variants without mutating the checked-in `config/project.yaml`.
Promoting a winning sweep child applies that child's scalar parameters to the base agent, keeps the base agent name unchanged, and promotes the base agent to paper mode.

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
Live promotion updates only the matching agent's `mode` to `live`. Set `deployment.mode: live` or pass `deploy --mode live` when generating a live deployment bundle.

Docker Compose and Render deployment bundles can be generated with:

```bash
poetry run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target docker-compose
poetry run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target docker-compose --mode live
poetry run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target render -o deployments/breakout-render
poetry run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target render --mode live -o deployments/breakout-render-live
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
    replay:
      enabled: true
      pace_delay_ms: 0
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
    execution:
      backend: "simulated"
    model:
      path: "models/promoted/aapl_daily_classifier"
    context:
      features: ["rsi_14"]
      positions: true
      risk_state: true
    risk:
      max_position_pct: 0.05

sweeps:
  - name: "rsi_threshold_grid"
    kind: "agent_backtest"
    agent: "paper_momentum"
    parameters:
      - path: "risk.max_position_pct"
        values: [0.03, 0.05, 0.07]

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
    execution:
      backend: "simulated"
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

SMA crossover rule-agent example:

```yaml
agents:
  - name: "sma_trend"
    kind: "rule"
    mode: "paper"
    execution:
      backend: "simulated"
    rule:
      preset: "sma_crossover"
      fast_feature: "sma_20"
      slow_feature: "sma_50"
    context:
      features: ["sma_20", "sma_50"]
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

Realtime-required fields:

- `enabled: true`
- `provider`
- `websocket_url`
- `symbols`
- `channels`

General runtime fields:

- `auth_method`
- `buffer_size`
- `reconnect_attempts`
- `health_check_interval`

Replay fields:

- `replay.enabled`
- `replay.start_date`
- `replay.end_date`
- `replay.pace_delay_ms`

Optional passthrough sections:

- `monitoring`
- `thresholds`
- `alerts`
- `metrics`
- `api`
- `rate_limit`
- `circuit_breaker`

These values are compiled into the runtime streaming YAML consumed by the current gateway.

Happy-path behavior:

- if `data.streaming.replay.enabled: true`, then `quanttradeai agent run --mode paper` replays historical OHLCV bars instead of connecting to a real-time websocket
- replay dates resolve from `replay.start_date` and `replay.end_date`, then `data.test_start` and `data.test_end`, then `data.start_date` and `data.end_date`
- local replay-backed paper runs may omit `provider` and `websocket_url`
- `quanttradeai agent run --mode live` and `quanttradeai deploy` still require real-time streaming fields

The default agent templates keep both the replay block and the real-time streaming block so the same project can move from local paper runs into later live promotion or paper deployment without restructuring the config.

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

Every agent can optionally define:

- `execution.backend`

Current execution backend support:

- `simulated`: default local execution path for backtest, replay-backed paper, and existing live simulations
- `alpaca`: real broker submission path for happy-path paper/live runs with Alpaca market data and broker credentials

Validation rules for `execution.backend: alpaca`:

- only supported when the agent itself is configured with `mode: paper` or `mode: live`
- requires `data.streaming.enabled: true`
- requires `data.streaming.provider: alpaca`
- rejects replay-backed paper mode
- warns during `validate` when `ALPACA_API_KEY` or `ALPACA_API_SECRET` are missing from the current environment

Broker-backed paper/live runs add:

- `summary.json` fields: `execution_backend`, `broker_provider`
- enriched `executions.jsonl` records with broker order IDs, statuses, fill quantities, and fill timestamps
- `broker_account_start.json`, `broker_account_end.json`, `broker_positions_start.json`, and `broker_positions_end.json`

#### `rule` agents

Required fields for all rule agents:

- `name`
- `kind: rule`
- `rule.preset`

Required fields for `rule.preset: rsi_threshold`:

- `rule.feature`
- `rule.buy_below`
- `rule.sell_above`

Required fields for `rule.preset: sma_crossover`:

- `rule.fast_feature`
- `rule.slow_feature`

Current built-in preset support:

- `rsi_threshold`
- `sma_crossover`

Validation checks that:

- the `rule` block exists for rule agents
- for `rsi_threshold`, `rule.feature` exists in `features.definitions`
- for `rsi_threshold`, `rule.feature` is also listed in `context.features`
- for `rsi_threshold`, `rule.buy_below < rule.sell_above`
- for `sma_crossover`, `rule.fast_feature` and `rule.slow_feature` exist in `features.definitions`
- for `sma_crossover`, both crossover features are listed in `context.features`
- for `sma_crossover`, both crossover features resolve to generated `sma_<period>` columns

Backtest mode:

- evaluates the configured feature on each completed bar
- writes `summary.json`, `metrics.json`, `equity_curve.csv`, `decisions.jsonl`, and `ledger.csv` when trades are present
- also writes compiled runtime config files and per-symbol coverage and metrics files under the run directory
- does not emit prompt sample artifacts because no LLM is involved

Paper mode:

- compiles `data.streaming` into a runtime streaming config
- warm-starts with recent historical OHLCV before streaming begins
- replays historical OHLCV deterministically when `data.streaming.replay.enabled` is true
- evaluates the configured rule on each completed streaming bar
- writes `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, and `runtime_streaming_config.yaml`
- writes `replay_manifest.json` and records `paper_source: replay` in `summary.json` when replay is enabled
- when `execution.backend: alpaca`, also writes broker account and position snapshots plus broker-enriched execution records
- does not emit `prompt_samples.json`

Live mode:

- requires the agent itself to already be configured with `mode: live`
- rejects `--skip-validation`
- compiles `data.streaming`, top-level `risk`, and top-level `position_manager` into runtime YAML snapshots inside the run directory
- writes `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, `runtime_streaming_config.yaml`, `runtime_risk_config.yaml`, and `runtime_position_manager_config.yaml`
- when `execution.backend: alpaca`, also writes broker account and position snapshots plus broker-enriched execution records

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
- warm-starts the model with historical bars so the first replay/live bar can infer immediately on the happy path
- runs the existing paper/live engine with the compiled feature config
- writes `summary.json`, `metrics.json`, `executions.jsonl`, and `runtime_streaming_config.yaml`
- writes `replay_manifest.json` and records `paper_source: replay` in `summary.json` when replay is enabled

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

Prompt context for `llm` and `hybrid` agents supports these blocks:

- `market_data`
- `features`
- `model_signals`
- `positions`
- `risk_state`
- `orders`
- `memory`
- `news`
- `notes`

`orders`, `memory`, and `news` accept either boolean shorthand or an object with `enabled` plus a cap:

- `orders: true` means `max_entries: 5`
- `memory: true` means `max_entries: 5`
- `news: true` means `max_items: 5`

`notes` accepts either boolean shorthand or an object with `enabled` and `file`:

- `notes: true` resolves to `notes/<agent_name>.md`
- the resolved notes file must exist and contain non-empty text

Validation rules:

- `context.news` requires top-level `news.enabled: true`
- `context.notes` fails validation when the resolved file is missing or empty

Runtime behavior:

- `orders.recent_orders` includes recent executions for the current symbol with `timestamp`, `action`, `qty`, `price`, and `status`
- `memory.recent_decisions` includes recent decisions for the current symbol with `timestamp`, `action`, `reason`, `execution_status`, and `target_position_after`
- `news.headlines` uses recent non-empty `text` rows already attached to the symbol history, newest first, deduped, capped
- `notes` is loaded once per run as `{path, content}`

Example:

```yaml
news:
  enabled: true
  provider: "yfinance"
  lookback_days: 7

agents:
  - name: "breakout_gpt"
    kind: "llm"
    mode: "paper"
    llm:
      provider: "openai"
      model: "gpt-5.3"
      prompt_file: "prompts/breakout.md"
    context:
      market_data:
        enabled: true
        timeframe: "1d"
        lookback_bars: 20
      features: ["rsi_14"]
      positions: true
      orders:
        enabled: true
        max_entries: 5
      memory: true
      news:
        enabled: true
        max_items: 5
      notes:
        enabled: true
        file: "notes/breakout_gpt.md"
      risk_state: true
```

Paper mode:

- compiles `data.streaming` into a runtime streaming config
- warm-starts with recent historical OHLCV before streaming begins
- replays historical OHLCV deterministically when `data.streaming.replay.enabled` is true
- aggregates streaming messages into completed bars before invoking the agent
- writes `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, `prompt_samples.json`, and `runtime_streaming_config.yaml`
- writes `replay_manifest.json` and records `paper_source: replay` in `summary.json` when replay is enabled

Live mode:

- requires `mode: live` in the agent config before `quanttradeai agent run --mode live` is allowed
- rejects `--skip-validation`
- compiles runtime streaming, risk, and position-manager YAML snapshots from `project.yaml`
- writes `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, `prompt_samples.json`, `runtime_streaming_config.yaml`, `runtime_risk_config.yaml`, and `runtime_position_manager_config.yaml`

### `deployment`

Deployment metadata for the canonical project workflow.

Current happy-path support:

- `target: "local"`
- `target: "docker-compose"`
- `target: "render"`
- `mode: "paper"` or `mode: "live"`

Example:

```yaml
deployment:
  target: "docker-compose"
  mode: "paper"
```

Behavior:

- `quanttradeai promote --run research/<run_id> -c config/project.yaml` copies trained model artifacts into stable `models/...` destinations and writes `promotion_manifest.json`
- `quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml` updates `deployment.mode` to `paper` when promoting a successful agent backtest run
- `quanttradeai deploy --agent <name> -c config/project.yaml --target local|docker-compose|render` generates a bundle under `reports/deployments/<agent>/<timestamp>/` unless `-o` is provided
- generated paper bundles force `data.streaming.replay.enabled: false` in `resolved_project_config.yaml`
- live bundles keep replay settings unchanged in `resolved_project_config.yaml`
- deployment generation fails if the project does not include the real-time `provider`, `websocket_url`, and `channels` required for deployment
- live deployment also requires the target agent to already be configured with `mode: live`, plus valid top-level `risk` and `position_manager` sections
- local bundles include `run.py`; Docker Compose bundles include `docker-compose.yml`; Render bundles include `render.yaml` plus `assets/` for selected-agent prompt, notes, and model files
- generated bundle services run `quanttradeai agent run --agent <name> -c config/project.yaml --mode <bundle-mode>`
- Render output must stay inside the project root because the generated Blueprint uses repo-relative Docker paths

### `sweeps`

Optional backtest-only parameter sweeps defined in the canonical project config.

Supported MVP shape:

```yaml
sweeps:
  - name: "rsi_threshold_grid"
    kind: "agent_backtest"
    agent: "rsi_reversion"
    parameters:
      - path: "rule.buy_below"
        values: [25.0, 30.0]
      - path: "rule.sell_above"
        values: [70.0, 75.0]
```

Rules:

- `kind` currently supports only `agent_backtest`
- `agent` must reference an existing project-defined agent
- `parameters[].path` is a dotted path relative to that agent, not the whole project config
- each path must resolve to an existing scalar leaf
- sweeps may not modify `name`, `kind`, or `mode`
- variants expand as a Cartesian product in the order parameters are declared
- each sweep child run records a `promote_command` in batch `results.json`
- promoting a successful sweep child materializes its parameter values into the base agent in `config/project.yaml` and sets the base agent to `mode: paper`
- sweep child runs cannot promote directly to live; run the materialized paper agent first, then promote that paper run to live

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

Agent promotion updates `config/project.yaml` in place. Backtest-to-paper promotion also updates `deployment.mode` to `paper`; paper-to-live promotion changes only the matching agent's mode, so live deployment can either use `deploy --mode live` explicitly or a separately configured `deployment.mode: live`.

Backtest runs generated by `quanttradeai agent run --sweep <name>` can be promoted when they complete successfully. Promotion materializes the winning scalar sweep parameters into the base agent, keeps the base agent name unchanged, and moves that base agent to paper mode. Sweep child runs cannot promote directly to live.

### `quanttradeai agent run --mode backtest`

Writes under `runs/agent/backtest/<timestamp>_<agent>/`.

### `quanttradeai agent run --sweep <name> --mode backtest`

Writes a batch directory under `runs/agent/batches/<timestamp>_<project>_<sweep>_backtest/` and one normal child backtest run under `runs/agent/backtest/...` for each expanded variant.

The sweep batch also writes `summary.json`, `experiment_brief.json`, and `experiment_brief.md`. The brief uses the winning child's existing `promote_command` and includes the follow-up paper command for the base agent.

### `quanttradeai agent run --mode paper`

Writes under `runs/agent/paper/<timestamp>_<agent>/`.

For `llm` and `hybrid` paper runs, the run directory includes both `decisions.jsonl` and `executions.jsonl` so prompt-driven decisions and actual paper executions can be audited separately.

When replay is enabled, the run directory also includes `replay_manifest.json`, and `summary.json` records `paper_source: replay`.

### `quanttradeai agent run --all --mode paper`

Writes a batch directory under `runs/agent/batches/<timestamp>_<project>_paper/` and one normal child paper run under `runs/agent/paper/...` for each enumerated agent.

Batch scoreboards for paper mode sort by `total_pnl`. The experiment brief recommends the live-promotion command for the top successful paper run.

### `quanttradeai agent run --all --mode live`

Writes a batch directory under `runs/agent/batches/<timestamp>_<project>_live/` and one normal child live run under `runs/agent/live/...` for each enumerated agent.

Live batches require `--acknowledge-live <project.name>`, reject `--skip-validation`, and fail fast unless every configured agent already has `mode: live`.

Batch scoreboards for live mode sort by `total_pnl`. The experiment brief points to metrics, executions, decisions, and logs for inspection instead of producing a promotion command.

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
