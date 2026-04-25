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
poetry run quanttradeai runs list --scoreboard --sort-by net_sharpe
poetry run quanttradeai runs list --compare research/<run_id_a> --compare research/<run_id_b>
```

This path gives you:

- one canonical project config
- resolved-config validation output
- a full research run with metrics and artifacts
- standardized run records under `runs/research/...`
- a scoreboard for ranking local runs and a compare flow for inspecting the shortlisted winners

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

### Generate A Deployment Bundle

```bash
poetry run quanttradeai deploy --agent paper_momentum -c config/project.yaml --target local
poetry run quanttradeai deploy --agent paper_momentum -c config/project.yaml --target docker-compose
```

Generated local and Docker Compose deployment bundles are still real-time paper deployments. QuantTradeAI disables replay in the emitted `resolved_project_config.yaml` and requires the normal provider and websocket settings to be present in the source project config.

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

Deployment bundles for project-defined paper agents are written under `reports/deployments/<agent>/<timestamp>/`. Use `--target local` for a Python runner bundle or `--target docker-compose` for a Compose bundle.

## Workflow 4: Multi-Agent Batches

Use this when one `config/project.yaml` already defines several agents and you want one local batch run across all of them.

```bash
poetry run quanttradeai agent run --all -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --all -c config/project.yaml --mode backtest --max-concurrency 4
poetry run quanttradeai agent run --all -c config/project.yaml --mode paper
poetry run quanttradeai agent run --all -c config/project.yaml --mode paper --max-concurrency 4
poetry run quanttradeai agent run --all -c config/project.yaml --mode live --acknowledge-live <project_name>
```

This workflow:

- validates the project before enumeration
- runs every configured agent through the existing backtest, paper, or live path
- preserves the normal child runs under `runs/agent/backtest/...`, `runs/agent/paper/...`, or `runs/agent/live/...`
- adds batch-level artifacts under `runs/agent/batches/<timestamp>_<project>_<mode>/`

Batch artifacts include:

- `batch_manifest.json`
- `results.json`
- `scoreboard.json`
- `scoreboard.txt`

Backtest batches rank by `net_sharpe`. Paper and live batches rank by `total_pnl`. Live batches require every configured agent to already have `mode: live` and require `--acknowledge-live` to match `project.name`.

## Standalone Utility Commands

These commands still exist for lower-level workflows that do not yet have a project-based replacement:

```bash
poetry run quanttradeai fetch-data -c config/model_config.yaml
poetry run quanttradeai evaluate -m <model_dir> -c config/model_config.yaml
poetry run quanttradeai backtest -c config/backtest_config.yaml
```

Important boundary:

- project-defined paper agents default to replay from `config/project.yaml`
- deployment bundles and live agents use real-time runtime YAML snapshots compiled from `config/project.yaml`
- the product happy path is still `init` -> `validate` -> `research run` or `agent run`

## Where To Go Next

- [Project YAML](configuration/project-yaml.md)
- [Generated Runtime Files](configuration/live-runtime-files.md)
- [Quick Reference](quick-reference.md)
- [Roadmap](../roadmap.md)
