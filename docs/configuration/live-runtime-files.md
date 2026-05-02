# Generated Runtime Files

QuantTradeAI generates runtime YAML snapshots from `config/project.yaml` so every run records the exact config that executed.

These files are artifacts, not inputs you maintain by hand.

## Research Runs

`quanttradeai research run -c config/project.yaml` writes:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `runtime_backtest_config.yaml`
- `summary.json`
- `metrics.json`

Use these files to inspect the compiled research settings that were passed into the lower-level training and backtest pipeline.

## Agent Backtest Runs

`quanttradeai agent run --mode backtest` writes:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `runtime_backtest_config.yaml`
- `summary.json`
- `metrics.json`
- `decisions.jsonl`

LLM and hybrid backtests also write `prompt_samples.json`.

## Agent Paper Runs

`quanttradeai agent run --mode paper` writes:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `runtime_streaming_config.yaml`
- `summary.json`
- `metrics.json`
- `decisions.jsonl`
- `executions.jsonl`

When replay is enabled, paper runs also write `replay_manifest.json`.

LLM and hybrid paper runs also write `prompt_samples.json`.

## Agent Live Runs

`quanttradeai agent run --mode live` writes:

- `resolved_project_config.yaml`
- `runtime_streaming_config.yaml`
- `runtime_risk_config.yaml`
- `runtime_position_manager_config.yaml`
- `summary.json`
- `metrics.json`
- `decisions.jsonl`
- `executions.jsonl`

LLM and hybrid live runs also write `prompt_samples.json`.

## Batch Runs

`quanttradeai agent run --all` and `quanttradeai agent run --sweep` write batch-level artifacts under `runs/agent/batches/...`, including:

- `resolved_project_config.yaml`
- `batch_manifest.json`
- `results.json`
- `scoreboard.json`
- `scoreboard.txt`

Child runs keep their normal per-run runtime YAML snapshots inside their own run directories. Live batches require `--acknowledge-live <project.name>` and preserve normal child runs under `runs/agent/live/...`.

## Deployment Bundles

`quanttradeai deploy --agent <name> -c config/project.yaml --target local|docker-compose|render` generates a deployment bundle from the same canonical config. The bundle includes a deployment manifest plus emitted runtime config files for the selected agent and mode.

Render bundles add a `render.yaml` Blueprint for a Docker-backed background worker and copy selected-agent prompt, notes, and model assets into `assets/` so the worker image can run from the emitted `resolved_project_config.yaml`.

## Utility Command Boundary

Standalone `fetch-data`, `evaluate`, and `backtest` still use their older single-purpose YAML files. They are utility commands, not the primary product workflow.
