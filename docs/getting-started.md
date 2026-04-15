# Getting Started

This guide covers the shortest working paths through QuantTradeAI.

## Install

```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
poetry install --with dev
```

## Workflow 1: Research From `project.yaml`

```bash
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai runs list
```

This path gives you:

- one canonical project config
- resolved-config validation output
- a full research run with metrics and artifacts
- standardized run records under `runs/research/...`

To promote a successful research run into the stable model path used by the `model-agent` and `hybrid` templates:

```bash
poetry run quanttradeai promote --run research/<run_id> -c config/project.yaml
```

## Workflow 2: Model Agent From `project.yaml`

```bash
poetry run quanttradeai init --template model-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
```

The model-agent template creates:

- a canonical `config/project.yaml`
- a placeholder model artifact directory at `models/promoted/aapl_daily_classifier/`
- a replay-enabled `data.streaming` block for local paper runs
- top-level `risk` and `position_manager` defaults for later live promotion

Replace the placeholder model directory with a promoted research model artifact or another compatible saved model before running the agent.

### Backtest The Agent

```bash
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode backtest
```

### Promote And Run The Same Agent In Paper Mode

```bash
poetry run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode paper
```

Local paper mode uses deterministic historical replay by default. If you leave `data.streaming.replay.start_date` and `end_date` unset, QuantTradeAI resolves the replay window from `data.test_start` and `data.test_end`, then falls back to `data.start_date` and `data.end_date`.

### Promote The Same Agent To Live

```bash
poetry run quanttradeai promote --run agent/paper/<run_id> -c config/project.yaml --to live --acknowledge-live paper_momentum
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode live
```

### Generate A Docker Compose Bundle

```bash
poetry run quanttradeai deploy --agent paper_momentum -c config/project.yaml --target docker-compose
```

Generated deployment bundles are still real-time paper deployments. QuantTradeAI disables replay in the emitted `resolved_project_config.yaml` and requires the normal provider and websocket settings to be present in the source project config.

Paper and live runs write standardized artifacts under `runs/agent/paper/...` and `runs/agent/live/...`, including:

- `summary.json`
- `metrics.json`
- `executions.jsonl`
- compiled runtime YAML snapshots

Replay-backed paper runs also write `replay_manifest.json`.

Live runs also write compiled `runtime_risk_config.yaml` and `runtime_position_manager_config.yaml`.

## Workflow 3: LLM Or Hybrid Agent

LLM and hybrid agents are supported in backtest, paper, and live mode from `project.yaml`.

```bash
poetry run quanttradeai init --template llm-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
poetry run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode paper
poetry run quanttradeai promote --run agent/paper/<run_id> -c config/project.yaml --to live --acknowledge-live breakout_gpt
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode live
```

Hybrid projects use the same pattern:

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

The hybrid template already points `model_signal_sources` at `models/promoted/aapl_daily_classifier`, so the happy path does not require editing timestamped experiment directories by hand.

Deployment bundles for project-defined paper agents are written under `reports/deployments/<agent>/<timestamp>/`.

## Workflow 4: Multi-Agent Backtest Batch

Use this when one `config/project.yaml` already defines several agents and you want one local batch run across all of them.

```bash
poetry run quanttradeai agent run --all -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --all -c config/project.yaml --mode backtest --max-concurrency 4
```

This workflow:

- validates the project before enumeration
- runs every configured agent through the existing backtest path
- preserves the normal child runs under `runs/agent/backtest/...`
- adds batch-level artifacts under `runs/agent/batches/<timestamp>_<project>_backtest/`

Batch artifacts include:

- `batch_manifest.json`
- `results.json`
- `scoreboard.json`
- `scoreboard.txt`

The batch workflow is backtest-only in this release. Paper and live still run one agent at a time.

## Legacy Runtime Workflows

These remain supported:

```bash
poetry run quanttradeai train -c config/model_config.yaml
poetry run quanttradeai evaluate -m <model_dir> -c config/model_config.yaml
poetry run quanttradeai backtest-model -m <model_dir> -c config/model_config.yaml -b config/backtest_config.yaml
poetry run quanttradeai live-trade -m <model_dir> -c config/model_config.yaml -s config/streaming.yaml
```

Important boundary:

- project-defined paper agents default to replay from `config/project.yaml`
- deployment bundles and live agents still use real-time streaming compiled from `config/project.yaml`
- `live-trade` still uses the legacy runtime YAML files directly

## Where To Go Next

- [Project YAML](configuration/project-yaml.md)
- [Runtime and Live Trading Configs](configuration/live-runtime-files.md)
- [Quick Reference](quick-reference.md)
- [Roadmap](../roadmap.md)
