# Artifacts

Every meaningful QuantTradeAI run writes structured files to disk so humans and coding agents can verify what ran, compare results, and make decisions from artifacts instead of terminal logs.

## Start here

For a single run, inspect artifacts in this order:

1. `summary.json`
2. `summary.json.run_result`
3. `metrics.json`
4. `resolved_project_config.yaml`

For batch and sweep runs, inspect artifacts in this order:

1. Batch `summary.json`
2. Batch `scoreboard.json`
3. Batch `results.json`
4. Top child run `summary.json`
5. Top child run `metrics.json`
6. Top child run `resolved_project_config.yaml`

`summary.json` tells you whether the run completed, where the important files are, and what high-level status QuantTradeAI recorded. `summary.json.run_result` is the compact decision field for winner and failure analysis. `metrics.json` contains the run metrics. `resolved_project_config.yaml` shows the final project configuration that actually executed.

## Core artifact types

| Artifact | Created by | Purpose | Inspect when |
| --- | --- | --- | --- |
| `summary.json` | `research run`, `agent run`, batch and sweep commands | Durable run record with status, run metadata, artifact paths, and compact result fields. | First, for any run or batch. |
| `summary.json.run_result` | Completion-oriented `research run`, `agent run`, batch and sweep commands | Compact result analysis inside `summary.json`, including ranked candidates and failures when available. | Making a quick pass/fail or winner decision. |
| `metrics.json` | Research runs and agent runs | Model, backtest, paper, or live metrics for a single run. | Comparing the actual performance of one run. |
| `scoreboard.json` | Research sweep, multi-agent batch, and agent sweep batch commands | Ranked batch-level scoreboard. | Choosing the leading candidate from a batch. |
| `results.json` | Research sweep, multi-agent batch, and agent sweep batch commands | Batch child results, run IDs, parameters, statuses, and promotion details when applicable. | Mapping a scoreboard row back to the child run and configuration variant. |
| `resolved_project_config.yaml` | Project-defined research, agent, and deployment commands | The fully resolved `config/project.yaml` used for the run or bundle. | Verifying what actually ran before trusting metrics or comparing runs. |
| `runtime_model_config.yaml` | Research and agent commands | Runtime model/data configuration compiled from project config. | Debugging model inputs, data windows, symbols, or serving/training consistency. |
| `runtime_features_config.yaml` | Research and agent commands | Runtime feature pipeline configuration compiled from project config. | Checking feature generation, labels, and training/serving consistency. |
| `runtime_backtest_config.yaml` | Research runs and agent backtests | Runtime backtest configuration compiled from project config. | Checking execution costs, backtest rules, or evaluation settings. |
| `runtime_streaming_config.yaml` | Agent paper and live runs | Runtime streaming configuration for replay, paper, or live execution. | Debugging paper/live data source behavior. |
| `runtime_risk_config.yaml` | Agent live runs | Runtime risk controls compiled for live execution. | Reviewing live-mode risk settings. |
| `runtime_position_manager_config.yaml` | Agent live runs | Runtime position manager settings compiled for live execution. | Reviewing live-mode position sizing and portfolio behavior. |
| `decisions.jsonl` | Agent backtest, paper, and live runs | Line-delimited decision records. | Auditing decision-level behavior. |
| `executions.jsonl` | Agent paper and live runs | Line-delimited paper or live execution records. | Auditing orders, fills, execution statuses, or paper/live behavior. |
| `prompt_samples.json` | LLM and hybrid agent runs | Sample prompts and prompt context captured for audit/debugging. | Debugging LLM or hybrid agent behavior. |
| `replay_manifest.json` | Replay-backed paper runs | Replay source and window metadata. | Verifying deterministic paper-mode replay inputs. |
| `deployment_manifest.json` | Deployment bundle generation | Machine-readable summary of a generated deployment bundle. | Validating what a deployment bundle contains. |

## Artifacts by workflow

## Research run

Path:
`runs/research/<timestamp>_<project>/`

Artifacts:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `runtime_backtest_config.yaml`
- `summary.json`
- `metrics.json`
- `backtest_summary.json`

Expected outcome:
A completed research run with resolved config, model/evaluation metrics, backtest summary, and promotion-ready artifacts if successful.

## Research sweep batch

Path:
`runs/research/batches/<timestamp>_<project>_<sweep>/`

Artifacts:

- `summary.json`
- `results.json`
- `scoreboard.json`

Expected outcome:
A batch-level comparison of many research variants.

## Agent backtest

Path:
`runs/agent/backtest/<timestamp>_<agent>/`

Artifacts:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `summary.json`
- `metrics.json`
- `decisions.jsonl`

Expected outcome:
A single agent backtest with metrics and decision records.

## Agent paper

Path:
`runs/agent/paper/<timestamp>_<agent>/`

Artifacts:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `runtime_streaming_config.yaml`
- `summary.json`
- `metrics.json`
- `executions.jsonl`
- `decisions.jsonl` for rule, LLM, and hybrid agents
- `prompt_samples.json` for LLM and hybrid agents
- `replay_manifest.json` when replay is enabled

Expected outcome:
A replay-backed or paper execution run with executions, decisions, metrics, and runtime config snapshots.

## Agent live

Path:
`runs/agent/live/<timestamp>_<agent>/`

Artifacts:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `runtime_streaming_config.yaml`
- `summary.json`
- `metrics.json`
- `executions.jsonl`
- `decisions.jsonl` for rule, LLM, and hybrid agents
- `prompt_samples.json` for LLM and hybrid agents
- `runtime_risk_config.yaml`
- `runtime_position_manager_config.yaml`

Expected outcome:
A live-mode run record. Live mode requires explicit human approval.

## Multi-agent batch

Path:
`runs/agent/batches/<timestamp>_<project>_<mode>/`

Artifacts:

- `summary.json`
- `results.json`
- `scoreboard.json`

Expected outcome:
Batch-level ranking across multiple configured agents.

## Agent sweep batch

Path:
`runs/agent/batches/<timestamp>_<project>_<sweep>_backtest/`

Artifacts:

- `summary.json`
- `results.json`
- `scoreboard.json`

Expected outcome:
Ranking of parameter variants for one base agent.

## Deployment bundles

Path:
`reports/deployments/<agent>/<timestamp>/`

Expected outcome:
A deployment-ready bundle. Generating a bundle is not the same as live trading.

### Local

Artifacts:

- `run.py`
- `.env.example`
- `README.md`
- `resolved_project_config.yaml`
- `deployment_manifest.json`

### Docker Compose

Artifacts:

- `docker-compose.yml`
- `Dockerfile`
- `.env.example`
- `README.md`
- `resolved_project_config.yaml`
- `deployment_manifest.json`

### Render

Artifacts:

- `render.yaml`
- `Dockerfile`
- `assets/`
- `.env.example`
- `README.md`
- `resolved_project_config.yaml`
- `deployment_manifest.json`

## How coding agents should use artifacts

- Do not declare a winner from terminal output alone.
- Check run status before trusting metrics.
- Use `summary.json.run_result` for compact decision-making.
- Use `scoreboard.json` for rankings.
- Use `results.json` to map candidates to run IDs and parameters.
- Use `resolved_project_config.yaml` to verify what actually ran.
- Read JSONL files only when decision/execution-level inspection is needed.
- Inspect `prompt_samples.json` only when debugging LLM/hybrid behavior.

## Common mistakes

- Confusing parent batch artifacts with child run artifacts.
- Treating paper success as live approval.
- Comparing runs without checking resolved configs.
- Trusting one metric without checking drawdown, failures, and warnings.
- Reading huge JSONL files unnecessarily.

## Related docs

- [Getting Started](getting-started.md)
- [CLI](cli/)
- [Configuration](config/)
- [Examples](examples/)
