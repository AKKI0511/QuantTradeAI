---
name: strategy-experiments
description: Practical cookbook for using QuantTradeAI inside an installed user workspace to test strategy ideas with YAML-first backtests, run one configured trading agent, run all configured agents, run agent parameter sweeps, rank candidates with scoreboards, compare finalists, and inspect run artifacts before making recommendations. Use when Codex is already inside a QuantTradeAI workspace and needs to operate the `quanttradeai` CLI and `config/project.yaml` for strategy research, not when contributing to the QuantTradeAI source repository.
---

# Strategy Experiments

Use this skill inside a user-created QuantTradeAI workspace, usually initialized with `quanttradeai init <workspace>`. Treat the installed `quanttradeai` CLI and the workspace YAML/artifacts as the source of truth. Do not read local QuantTradeAI source files or local docs as part of this workflow.

Keep strategy experiments in `backtest` mode unless the user explicitly moves beyond research. Do not use broker credentials, run paper/live mode, promote, or deploy as part of this skill.

## Quick Orientation

1. Check CLI availability only if needed:

   ```bash
   quanttradeai --help
   ```

2. Locate the project config. Prefer the user-provided path; otherwise use:

   ```text
   config/project.yaml
   ```

3. Inspect the config before editing. Identify:

   - `data`: symbols, dates, timeframe, test window, cache settings.
   - `features`: feature definitions used by agents.
   - `agents`: configured strategies and their rule/model/LLM/hybrid settings.
   - `sweeps`: named `agent_backtest` parameter grids.
   - Strategy/risk fields: rule thresholds, SMA feature names, `risk.max_position_pct`, `risk.max_portfolio_risk`, execution backend.

Prefer deterministic rule/model strategy experiments. Do not add broker-backed execution or credentials. If LLM/hybrid agents already exist, warn that backtests may call external model APIs unless the project is configured to avoid that.

## YAML Editing Rules

Use the smallest config edit that can answer the experiment question.

- Keep behavior config-driven in `config/project.yaml` or the user-provided `-c <config>`.
- Keep strategy names stable unless a new candidate is intentionally added.
- Use `execution.backend: simulated` for research backtests.
- Make sure rule features exist under `features.definitions` and are listed in the agent's `context.features`.
- For `rsi_threshold` agents, keep `rule.feature`, `rule.buy_below`, and `rule.sell_above`, with buy below sell.
- For `sma_crossover` agents, keep `rule.fast_feature` and `rule.slow_feature`, both present in `context.features`.
- Keep sweep paths pointed at existing scalar leaves on the selected agent, such as `rule.buy_below`, `rule.sell_above`, `rule.fast_feature`, `rule.slow_feature`, or `risk.max_position_pct`.
- Do not expect sweeps to mutate the source YAML. Sweeps create generated variants and child run artifacts.

Example agent sweep shape:

```yaml
sweeps:
  - name: rsi_threshold_grid
    kind: agent_backtest
    agent: rsi_reversion
    parameters:
      - path: rule.buy_below
        values: [25, 30]
      - path: rule.sell_above
        values: [70, 75]
      - path: risk.max_position_pct
        values: [0.03, 0.05]
```

## Validate Before Running

Validation confirms the config can be used and catches missing sections, invalid agent references, unsupported sweep paths, missing prompt/model files, and paper/live prerequisites.

```bash
quanttradeai validate -c config/project.yaml
```

If validation fails, fix the YAML and validate again before running backtests.

## Run One Agent Backtest

Use a single-agent backtest to test one configured strategy candidate.

```bash
quanttradeai agent run --agent <agent_name> -c config/project.yaml --mode backtest
```

Expected output path:

```text
runs/agent/backtest/<timestamp>_<agent_name>/
```

Inspect at least `summary.json`, `summary.json.run_result`, `metrics.json`, and `resolved_project_config.yaml` before drawing conclusions.

## Run All Configured Agents

Use an all-agent backtest to compare every configured agent in the project.

```bash
quanttradeai agent run --all -c config/project.yaml --mode backtest --max-concurrency <n>
```

Expected batch path:

```text
runs/agent/batches/<timestamp>_<project_name>_backtest/
```

The batch creates child runs under `runs/agent/backtest/...` and batch artifacts under `runs/agent/batches/...`.

## Run An Agent Parameter Sweep

Use an agent sweep to expand parameter variants for one base agent and rank the variants. Agent sweeps are backtest-focused.

```bash
quanttradeai agent run --sweep <sweep_name> -c config/project.yaml --mode backtest --max-concurrency <n>
```

Expected batch path:

```text
runs/agent/batches/<timestamp>_<project_name>_<sweep_name>_backtest/
```

Expected batch artifacts:

- `summary.json`: batch status, counts, warnings, and `run_result`.
- `scoreboard.json`: ranked records, normally sorted by `net_sharpe` for backtest batches.
- `results.json`: child run IDs, statuses, parameters, variant config paths, artifact paths, and failures.
- `variants/<variant>/project.yaml`: generated variant project configs for sweep children.

Child backtests are created under:

```text
runs/agent/backtest/<timestamp>_<variant_agent_name>/
```

## Rank And Compare Candidates

Use a scoreboard after single, all-agent, or sweep runs to identify candidates.

```bash
quanttradeai runs list --type agent --mode backtest --scoreboard --sort-by net_sharpe
```

Use explicit comparison for 2-4 finalists from the same run family:

```bash
quanttradeai runs list --compare agent/backtest/<run_a> --compare agent/backtest/<run_b> --sort-by net_sharpe
```

The scoreboard is a filter, not the final recommendation. Prefer `net_sharpe` for initial ranking, then check net PnL, max drawdown, decision/trade activity, warnings, and failures.

## Artifact Inspection Checklist

Before recommending a winner, inspect artifacts in this order:

1. Batch `summary.json`, especially `status`, `warnings`, and `run_result`.
2. Batch `scoreboard.json` for ranking and metric values.
3. Batch `results.json` to map ranked rows to child run IDs, parameters, statuses, and variant configs.
4. Top child `summary.json` and `summary.json.run_result`.
5. Top child `metrics.json`.
6. Top child `resolved_project_config.yaml`.

Do not recommend a winner from terminal output alone. Check that the winning child status is successful and that the resolved config matches the intended strategy, symbols, dates, feature definitions, and risk settings.

## Recommendation Standards

A useful final response should include:

- Config changes made.
- Commands run.
- Pass/fail status for validation and each run.
- Artifact paths inspected.
- Best strategy candidate, run ID, and parameters.
- Closest alternatives.
- Key metrics, especially net Sharpe, net PnL, max drawdown, decision/trade count, and failure count.
- Warnings, failed variants, inactive strategies, or overfitting concerns.
- Assumptions and the next research step.

Avoid saying a backtest winner is ready for paper, live, promotion, or deployment unless the user explicitly asks for the next stage.

## More Detail

For product documentation, use:

- https://akkijoshi.gitbook.io/quanttradeai/
- https://github.com/AKKI0511/QuantTradeAI/tree/main/docs
