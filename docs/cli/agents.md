# Agent Commands

`quanttradeai agent run` runs a project-defined trading agent from `config/project.yaml`. The coding agent uses this CLI command; the YAML agent is the trading strategy QuantTradeAI executes.

## Operator Agent Vs Trading Agent

| Term | Meaning |
|---|---|
| Operator or coding agent | The human or AI assistant invoking CLI commands, editing YAML, and inspecting artifacts. |
| Trading agent | A configured item in `agents[]` with `kind: rule`, `model`, `llm`, or `hybrid`. |

## When To Use

Use `agent run` when you want to execute one or more YAML-defined trading agents:

- `backtest`: replay the configured historical test window and write decision/backtest artifacts
- `paper`: run against replay or real-time streaming input, depending on project config
- `live`: run a live-configured agent with live runtime risk and position-manager prerequisites

Use `research run` instead when training model artifacts from the `research` section.

## Syntax

```bash
quanttradeai agent run [--agent NAME | --all | --sweep NAME] [OPTIONS]
```

Exactly one of `--agent`, `--all`, or `--sweep` must be provided.

```bash
quanttradeai agent run --agent rsi_reversion
quanttradeai agent run --agent rsi_reversion --mode paper
quanttradeai agent run --all --mode backtest --max-concurrency 2
quanttradeai agent run --sweep risk-grid --mode backtest
quanttradeai agent run --all --mode live --acknowledge-live strategy_lab
```

## Options

| Option | Default | Required | Description |
|---|---:|:---:|---|
| `--agent TEXT` | `None` | One of selection options | Run one agent by name from `agents[]`. |
| `--all` | `false` | One of selection options | Run every project-defined agent. Supports `backtest`, `paper`, and `live`. |
| `--sweep TEXT` | `None` | One of selection options | Run every expanded variant for the named agent sweep. Backtest mode only. |
| `-c, --config TEXT` | `config/project.yaml` | No | Path to the project config YAML. |
| `--mode TEXT` | `backtest` | No | Execution mode. Supported values are `backtest`, `paper`, and `live`. |
| `--skip-validation` | `false` | No | Skip data-quality validation before backtesting. Ignored with warnings for paper runs; not supported for live runs. |
| `--max-concurrency INTEGER` | `1` | No | Maximum concurrent child runs when using `--all` or `--sweep`. Must be at least `1`. |
| `--acknowledge-live TEXT` | `None` | Live batch only | Required for `--all --mode live`; value must exactly match `project.name`. |

## Selection Modes

| Selection | What happens |
|---|---|
| `--agent <name>` | Runs one configured agent. |
| `--all` | Runs all configured agents and writes a batch record. |
| `--sweep <name>` | Expands an `agent_backtest` sweep and runs each variant as a child backtest. |

The CLI enforces the exactly-one rule. Passing none or more than one selection option fails before the run starts.

## Execution Modes

| Mode | Supported agent kinds | Notes |
|---|---|---|
| `backtest` | `rule`, `model`, `llm`, `hybrid` | Uses historical data and writes backtest metrics, decisions, and equity artifacts. |
| `paper` | `rule`, `model`, `llm`, `hybrid` | Uses replay if `data.streaming.replay.enabled` resolves a replay window; otherwise uses real-time streaming. |
| `live` | `rule`, `model`, `llm`, `hybrid` | Requires the agent itself to be configured with `mode: live`; live batches require `--acknowledge-live <project.name>`. |

For non-live runs, the CLI requested `--mode` can differ from the agent's configured `mode`; the run continues and emits a warning. For live runs, the configured agent mode must already be `live`.

Paper mode does not imply broker-backed execution by itself. The execution backend is resolved from the agent config. The default backend is simulated unless an agent is configured for a broker-backed backend such as `alpaca`.

## Reads

```text
config/project.yaml
```

Depending on the agent, the run can also read:

- prompt files for `llm` and `hybrid` agents
- promoted model directories for `model` agents
- model signal source directories for `hybrid` agents
- notes files when notes context is enabled
- market data cache or provider data

## Writes / Artifacts

Single agent runs write under:

```text
runs/agent/backtest/<timestamp>_<agent_name>/
runs/agent/paper/<timestamp>_<agent_name>/
runs/agent/live/<timestamp>_<agent_name>/
```

Batch and sweep runs write under:

```text
runs/agent/batches/<timestamp>_<project_name>_<mode>/
runs/agent/batches/<timestamp>_<project_name>_<sweep_name>_<mode>/
```

Common run artifacts include:

| Artifact | Modes | Purpose |
|---|---|---|
| `summary.json` | all | Durable run record and compact run result. |
| `resolved_project_config.yaml` | all | Resolved project config snapshot. |
| `runtime_model_config.yaml` | all | Runtime model/data config. |
| `runtime_features_config.yaml` | all | Runtime feature config. |
| `metrics.json` | all successful runs | Scoreboard metrics. |
| `decisions.jsonl` | all successful runs when decisions exist | Agent decisions by timestamp/symbol. |
| `executions.jsonl` | paper/live | Streaming execution log. |
| `equity_curve.csv` | backtest | Aggregate equity curve. |
| `ledger.csv` | backtest when trades exist | Combined execution ledger. |
| `prompt_samples.json` | LLM/hybrid runs | Limited prompt/response samples. |
| `replay_manifest.json` | replay-backed paper | Replay split metadata. |
| `runtime_streaming_config.yaml` | paper/live | Runtime streaming config. |
| `runtime_risk_config.yaml` | live | Runtime risk config. |
| `runtime_position_manager_config.yaml` | live | Runtime position-manager config. |

Batch artifacts include `results.json`, `scoreboard.json`, `summary.json`, `resolved_project_config.yaml`, and per-child logs under `logs/`.

See [`docs/artifacts.md`](../artifacts.md) for artifact conventions.

## CLI Result

The command prints compact JSON. Single runs include the run id, status, run directory, run type, mode, name, and available metrics. Batch runs include counts and, when rankable, a `winner`.

## Examples

Backtest one agent:

```bash
quanttradeai agent run --agent rsi_reversion
```

Run one paper agent:

```bash
quanttradeai agent run --agent rsi_reversion --mode paper
```

Run all backtests:

```bash
quanttradeai agent run --all --mode backtest --max-concurrency 4
```

Run an agent sweep:

```bash
quanttradeai agent run --sweep risk-grid --mode backtest
```

Run all live agents after promotion:

```bash
quanttradeai agent run --all --mode live --acknowledge-live strategy_lab
```

Inspect the results:

```bash
quanttradeai runs list --type agent --mode backtest --scoreboard
```

## Expected Outcome

A successful run writes a durable `summary.json` and supporting artifacts under `runs/agent/...`. A failed child run in a batch is captured in the batch `results.json`, and failed single runs write a failed summary before exiting.

## Common Mistakes

| Mistake | Why it fails or misleads |
|---|---|
| Passing more than one of `--agent`, `--all`, and `--sweep` | The CLI requires exactly one selection mode. |
| Using `--sweep` with `--mode paper` or `--mode live` | Sweeps are backtest-only in the current implementation. |
| Missing live acknowledgement for `--all --mode live` | Live batches require `--acknowledge-live <project.name>`. |
| Passing `--acknowledge-live` with a single agent run | The flag is only accepted with `--all --mode live`. |
| Assuming paper mode is broker-backed by default | Paper uses simulated execution unless the agent config selects a broker-backed execution backend. |
| Not checking artifacts after a run | The compact JSON is intentionally sparse; inspect `summary.json`, `metrics.json`, and run-specific logs before promotion. |
| Using `--skip-validation` for live | Live runs reject `--skip-validation`. |

## Related Docs

- [`docs/config/`](../config/)
- [`docs/artifacts.md`](../artifacts.md)
