# Agents

The `agents` section defines trading agents that QuantTradeAI can backtest, run in paper mode, run in live mode, sweep, promote, and deploy.

> **Important distinction:** a coding agent/operator is Claude Code, Codex, or Cursor editing YAML and running CLI commands. A trading agent is a YAML-defined `rule`, `model`, `llm`, or `hybrid` strategy executed by QuantTradeAI.

## Used By

| Command | How it uses `agents` |
|---|---|
| `quanttradeai validate` | Validates agent shape, file paths, context references, streaming requirements, and live prerequisites. |
| `quanttradeai agent run --agent <name>` | Runs one configured agent. |
| `quanttradeai agent run --all` | Runs every configured agent as a batch. |
| `quanttradeai agent run --sweep <name>` | Expands an `agent_backtest` sweep and runs backtest variants. |
| `quanttradeai promote --run agent/<mode>/<run>` | Promotes successful agent backtest runs to paper or paper runs to live. |
| `quanttradeai deploy --agent <name>` | Generates a deployment bundle for one configured agent. |

## Supported Agent Kinds

| Kind | Purpose | Required fields | Common optional fields |
|---|---|---|---|
| `rule` | Deterministic built-in rules over generated features. | `name`, `kind`, `mode`, `rule`, `context.features` | `execution`, `risk`, `tools`, `context.positions`, `context.risk_state` |
| `model` | Uses a trained classifier artifact for signals. | `name`, `kind`, `mode`, `model.path` | `execution`, `risk`, `tools` |
| `llm` | Calls a LiteLLM-backed model with a prompt and deterministic context. | `name`, `kind`, `mode`, `llm`, prompt file | `context`, `tools`, `risk`, `execution` |
| `hybrid` | Combines LLM decisions with model signal context. | `name`, `kind`, `mode`, `llm` | `model_signal_sources`, `context.model_signals`, `tools`, `risk`, `execution` |

## Supported Fields

| Field | Type | Notes |
|---|---|---|
| `name` | string | Unique agent name used by CLI flags and run artifacts. |
| `kind` | enum | `rule`, `model`, `llm`, or `hybrid`. |
| `mode` | enum | `backtest`, `paper`, or `live`. Live CLI runs require the agent to already be configured with `mode: live`. |
| `execution.backend` | enum | `simulated` or `alpaca`. Defaults to `simulated`. |
| `tools` | list | Supported values: `get_quote`, `get_position`, `place_order`. These are passed as prompt context, not executed as coding-agent tools. |
| `risk` | mapping | Runtime sizing reads `max_position_pct` and `max_portfolio_risk`; other keys are available in `context.risk_state`. |

### Rule Block

```yaml
agents:
  - name: rsi_reversion
    kind: rule
    mode: paper
    rule:
      preset: rsi_threshold
      feature: rsi_14
      buy_below: 30
      sell_above: 70
    context:
      features: [rsi_14]
```

| Preset | Required fields | Validation |
|---|---|---|
| `rsi_threshold` | `feature`, `buy_below`, `sell_above` | `feature` must be listed in `context.features` and resolve to a scalar RSI value. |
| `sma_crossover` | `fast_feature`, `slow_feature` | Both features must be listed in `context.features`, be different, and resolve to generated SMA values. |

### Model Block

```yaml
agents:
  - name: paper_momentum
    kind: model
    mode: paper
    model:
      path: models/promoted/aapl_daily_classifier
```

`model.path` must resolve to an existing artifact directory or file at validation time. Docker Compose and Render deployment targets require model paths to resolve under `models/`.

### LLM Block

```yaml
agents:
  - name: breakout_gpt
    kind: llm
    mode: paper
    llm:
      provider: openai
      model: gpt-5.3
      prompt_file: prompts/breakout.md
      api_key_env_var: OPENAI_API_KEY
      extra:
        temperature: 0
```

| Field | Notes |
|---|---|
| `provider` | Required. Default API key env vars are known for `openai`, `anthropic`, and `huggingface`. |
| `model` | Required. Passed to LiteLLM as `<provider>/<model>` unless the model already contains `/`. |
| `prompt_file` | Required and must exist. Resolved relative to the project root. |
| `api_key_env_var` | Optional override for the provider API key environment variable. |
| `extra` | Optional mapping passed through to the LiteLLM completion call. |

LLM responses must be JSON with `action` set to `buy`, `sell`, or `hold`, and a non-empty `reason`.

### Hybrid Model Signals

```yaml
agents:
  - name: hybrid_swing_agent
    kind: hybrid
    mode: paper
    llm:
      provider: openai
      model: gpt-5.3
      prompt_file: prompts/hybrid_swing.md
    model_signal_sources:
      - name: aapl_daily_classifier
        path: models/promoted/aapl_daily_classifier
    context:
      model_signals: [aapl_daily_classifier]
```

Each `context.model_signals` entry must reference a `model_signal_sources[].name`, and each source path must exist.

## Context Options

| Field | Applies to | Notes |
|---|---|---|
| `context.market_data` | rule, llm, hybrid | Boolean or mapping with `enabled`, `timeframe`, and `lookback_bars`. |
| `context.features` | rule, llm, hybrid | Feature names from `features.definitions` to expose in decision context. |
| `context.model_signals` | hybrid | Names from `model_signal_sources`. |
| `context.positions` | rule, llm, hybrid | Adds target position state. |
| `context.risk_state` | rule, llm, hybrid | Adds decision count, current direction, and `agents[].risk`. |
| `context.orders` | llm, hybrid | Boolean or `{enabled, max_entries}` for recent executions. |
| `context.memory` | llm, hybrid | Boolean or `{enabled, max_entries}` for recent decisions. |
| `context.news` | llm, hybrid | Boolean or `{enabled, max_items}`. Requires top-level `news.enabled: true`. |
| `context.notes` | llm, hybrid | Boolean or `{enabled, file}`. Defaults to `notes/<agent_name>.md` and the file must exist and be non-empty. |

```yaml
news:
  enabled: true

agents:
  - name: breakout_gpt
    kind: llm
    mode: backtest
    llm:
      provider: openai
      model: gpt-5.3
      prompt_file: prompts/breakout.md
    context:
      market_data: {enabled: true, timeframe: 1d, lookback_bars: 20}
      features: [rsi_14]
      positions: true
      orders: {enabled: true, max_entries: 5}
      memory: {enabled: true, max_entries: 5}
      news: {enabled: true, max_items: 5}
      notes: {enabled: true, file: notes/breakout_gpt.md}
```

## Mode Behavior

| CLI mode | Behavior |
|---|---|
| `backtest` | Uses historical data and simulated executions. If the configured agent mode differs, QuantTradeAI warns and continues, except for live-specific restrictions. |
| `paper` | Requires streaming. Uses replay when `data.streaming.replay.enabled: true`; otherwise uses real-time streaming. `--skip-validation` is ignored. |
| `live` | Requires the agent to be configured with `mode: live`; `--skip-validation` is rejected. Requires real-time streaming, top-level `risk`, and `position_manager`. |

## Execution Backends

| Backend | Behavior |
|---|---|
| `simulated` | Default. Backtests and paper/live runs use local simulated fills. |
| `alpaca` | Paper/live broker-backed execution through Alpaca REST. Requires `data.streaming.enabled: true`, `data.streaming.provider: alpaca`, real-time paper mode, and `ALPACA_API_KEY` plus `ALPACA_API_SECRET`. |

Alpaca-backed paper mode submits orders to Alpaca paper trading. Alpaca-backed live mode submits orders to Alpaca live trading.

## CLI Selection

```bash
# One agent
quanttradeai agent run --agent rsi_reversion -c config/project.yaml --mode backtest

# Every configured agent
quanttradeai agent run --all -c config/project.yaml --mode paper --max-concurrency 2

# Sweep variants, backtest only
quanttradeai agent run --sweep rsi_threshold_grid -c config/project.yaml --mode backtest
```

`--agent`, `--all`, and `--sweep` are mutually exclusive. Choose exactly one.

## Expected Outcomes

Single agent runs write under `runs/agent/<mode>/...`:

- `summary.json`
- `metrics.json`
- `decisions.jsonl`
- `executions.jsonl` for paper/live runs
- `prompt_samples.json` for LLM and hybrid agents
- `equity_curve.csv` and optional `ledger.csv` for backtests
- runtime config snapshots
- `replay_manifest.json` for replay-backed paper mode
- broker account and position snapshots when `execution.backend: alpaca`

Batch runs write under `runs/agent/batches/...`:

- `summary.json`
- `results.json`
- `scoreboard.json`
- child run artifacts
- per-child stdout/stderr logs

## Common Mistakes

| Mistake | Result |
|---|---|
| Passing an agent name that does not match `agents[].name` | The command fails with an agent lookup error. |
| Requesting live mode while the agent is not configured `mode: live` | The live run or live deployment fails. |
| Missing an LLM prompt file or model artifact path | Validation fails. |
| Assuming an LLM trading agent is the coding agent/operator | LLM agents are YAML-defined trading strategies executed by QuantTradeAI. |
| Enabling `context.news` without `news.enabled: true` | Validation fails. |
| Using `execution.backend: alpaca` with replay-backed paper mode | Validation fails. |
