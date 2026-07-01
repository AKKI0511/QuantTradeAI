---
name: promote-deploy
description: Practical cookbook for safely moving successful QuantTradeAI results beyond research/backtest in an installed user workspace. Use when a user asks to promote a run, move a strategy/model/agent to paper, prepare deployment, generate a deployment bundle, deploy an agent, move from paper to live, run live, use broker-backed execution, or package a winning strategy for execution with `quanttradeai promote`, `agent run`, and `deploy` while respecting live-trading safety boundaries.
---

# Promote And Deploy

Use this skill inside a user-created QuantTradeAI workspace after a successful research, backtest, or paper run has already been selected. Treat the installed `quanttradeai` CLI, the project YAML, and generated artifacts as the source of truth.

Do not create new research experiments by default. Use `quanttradeai runs list` and artifact inspection only as verification steps before promotion or deployment.

## Safety Boundaries

Promotion is not deployment. Deployment bundle generation is not live trading.

Never run `--mode live` without explicit human approval. Never add `--acknowledge-live` unless the human clearly approved live. Never use broker credentials or broker-backed execution unless explicitly requested. Never treat paper success as live approval. Never promote to live just because a backtest or paper run succeeded.

If the user says "deploy", clarify whether they mean:

- generate a deployment bundle
- run a paper agent
- promote an agent to live
- start live trading
- enable broker-backed execution

If the instruction is ambiguous and could affect live or broker-backed execution, stop and clarify before acting.

## Decision Flow

1. Confirm the target run or agent.
2. Inspect enough artifacts to verify the target succeeded.
3. Confirm whether the user wants paper, a deployment bundle, or live.
4. For paper, promote and/or run paper if appropriate.
5. For a deployment bundle, generate the requested bundle target.
6. For live, require explicit approval and the exact acknowledgement before running live promotion or live batch commands.
7. Report what changed, what was generated, and what remains manual.

## Commands

Promote a successful run:

```bash
quanttradeai promote --run <run_id> -c <config>
```

Promote a successful paper agent run to live only with explicit approval:

```bash
quanttradeai promote --run <run_id> -c <config> --to live --acknowledge-live <agent_name>
```

Run a paper agent:

```bash
quanttradeai agent run --agent <agent_name> -c <config> --mode paper
```

Run a live agent only with explicit approval:

```bash
quanttradeai agent run --agent <agent_name> -c <config> --mode live
```

Run all live agents only with explicit approval and exact project acknowledgement:

```bash
quanttradeai agent run --all -c <config> --mode live --acknowledge-live <project_name>
```

Generate deployment bundles:

```bash
quanttradeai deploy --agent <agent_name> -c <config> --target local
quanttradeai deploy --agent <agent_name> -c <config> --target docker-compose
quanttradeai deploy --agent <agent_name> -c <config> --target render
```

Use `quanttradeai runs list` only to confirm run IDs, statuses, and artifact paths:

```bash
quanttradeai runs list
quanttradeai runs list --type agent --mode backtest --status success
quanttradeai runs list --type agent --mode paper --status success
```

## What Promotion Changes

Research run promotion copies trained symbol artifacts from the research experiment directory into stable paths from `research.promotion.targets[]`, usually under `models/promoted/...`, and writes `promotion_manifest.json` inside each promoted destination.

Agent backtest to paper promotion updates the selected agent to `mode: paper` and updates `deployment.mode: paper` when needed.

Agent sweep backtest promotion materializes the winning sweep parameters into the base agent, sets that base agent to `paper`, and updates `deployment.mode: paper` when needed.

Agent paper to live promotion updates the selected agent to `mode: live` only after live prerequisites pass and `--acknowledge-live <agent_name>` exactly matches the agent.

Research promotion does not support `--to live`. Agent backtests cannot be promoted directly to live; paper is the required intermediate stage.

## YAML Areas To Verify

Before paper promotion or paper runs, verify:

- `agents[]`: target agent exists and points to the intended model/rule/LLM/hybrid assets.
- `data.streaming.enabled`: true for paper agents.
- `agents[].execution.backend`: `simulated` unless broker-backed execution was explicitly requested.
- Promoted model paths exist when using model or hybrid agents.

Before live promotion, live runs, or live bundles, verify:

- Target agent is or will become `mode: live` through an explicit live promotion.
- `data.streaming` has real-time provider, websocket URL, channels, and replay disabled or irrelevant.
- Top-level `risk` exists and includes enabled drawdown protection.
- Top-level `position_manager` exists.
- `deployment.mode` and `deployment.target` match the intended bundle behavior.
- Broker-backed execution is intentional if `agents[].execution.backend: alpaca`.

Broker-backed Alpaca execution requires explicit approval plus `data.streaming.provider: alpaca`, real-time streaming, `agents[].execution.backend: alpaca`, and environment variables such as `ALPACA_API_KEY` and `ALPACA_API_SECRET`.

## Verification Artifacts

Before promotion or deployment, inspect enough artifacts to confirm the selected run is successful:

- `summary.json`: status, run ID, warnings, artifact paths.
- `summary.json.run_result` when present.
- `metrics.json`: paper/live/backtest metrics.
- `resolved_project_config.yaml`: exact config that ran.
- `executions.jsonl`: paper/live execution records when relevant.
- `backtest_summary.json`: research model backtest details when relevant.

After paper runs, expect:

```text
runs/agent/paper/<timestamp>_<agent_name>/
```

Inspect `summary.json`, `metrics.json`, `executions.jsonl`, `resolved_project_config.yaml`, and `replay_manifest.json` when replay was used.

After live runs, expect:

```text
runs/agent/live/<timestamp>_<agent_name>/
```

Inspect `summary.json`, `metrics.json`, `executions.jsonl`, `decisions.jsonl` when present, `runtime_streaming_config.yaml`, `runtime_risk_config.yaml`, `runtime_position_manager_config.yaml`, and `resolved_project_config.yaml`.

After deployment bundle generation, expect:

```text
reports/deployments/<agent_name>/<timestamp>/
```

Every deployment bundle should include `deployment_manifest.json`, `resolved_project_config.yaml`, `README.md`, and `.env.example`.

Target-specific files:

- `local`: `run.py`
- `docker-compose`: `docker-compose.yml`, `Dockerfile`
- `render`: `render.yaml`, `Dockerfile`, `assets/`

Read `deployment_manifest.json` to confirm `agent_name`, `target`, `mode`, `execution_backend`, `broker_provider`, generated artifact paths, warnings, safety requirements, environment variables, and `next_command`.

## Paper Workflow

Use this when the user wants to move a successful backtest or materialized sweep winner to paper.

1. Confirm the `agent/backtest/<run_id>` target.
2. Inspect `summary.json`, `metrics.json`, and `resolved_project_config.yaml`.
3. Run:

   ```bash
   quanttradeai promote --run <run_id> -c <config>
   ```

4. Review promotion output for changed fields and `next_command`.
5. Run paper only if the user requested it or the task clearly includes paper-stage checks:

   ```bash
   quanttradeai agent run --agent <agent_name> -c <config> --mode paper
   ```

6. Inspect the paper run artifacts before recommending any live step.

## Deployment Bundle Workflow

Use this when the user wants a package or bundle, not an active live run.

1. Confirm the agent and target: `local`, `docker-compose`, or `render`.
2. Verify the selected agent config and required assets exist.
3. For paper bundles, confirm real-time streaming fields are present. Paper deployment bundles disable replay in the generated bundle.
4. For live bundles, confirm the agent is already configured with `mode: live`.
5. Run the requested bundle command.
6. Inspect `deployment_manifest.json` and `resolved_project_config.yaml`.
7. Report the bundle path and `next_command` as manual unless the user explicitly asks to run it.

## Live Workflow

Use this only after the user explicitly approves live.

1. Confirm the paper run ID and target agent.
2. Inspect paper `summary.json`, `metrics.json`, `executions.jsonl`, and `resolved_project_config.yaml`.
3. Confirm exact acknowledgement text from the human: the `agent_name` for paper-to-live promotion.
4. Run:

   ```bash
   quanttradeai promote --run <run_id> -c <config> --to live --acknowledge-live <agent_name>
   ```

5. Verify config changes and live prerequisites.
6. Run a single live agent only if explicitly requested:

   ```bash
   quanttradeai agent run --agent <agent_name> -c <config> --mode live
   ```

7. Run all live agents only if explicitly requested and the human acknowledges the exact `project.name`:

   ```bash
   quanttradeai agent run --all -c <config> --mode live --acknowledge-live <project_name>
   ```

8. Inspect live artifacts immediately after the run.

## Reporting Standards

When reporting back, include:

- Run or agent selected.
- Artifacts checked before promotion or deployment.
- Commands run.
- Config changes made.
- Generated bundle path or paper/live run path.
- Safety assumptions and approvals used.
- Missing requirements or warnings.
- Next manual or automated step.

For deployment bundles, make clear that the bundle was generated and no trading started unless a run command was explicitly executed.

## More Detail

Use these references only as fallback:

- https://akkijoshi.gitbook.io/quanttradeai/
- https://github.com/AKKI0511/QuantTradeAI/tree/main/docs
