---
name: run-analysis
description: Practical cookbook for analyzing completed QuantTradeAI runs and artifacts in an installed user workspace. Use when a user asks to analyze results, compare runs, find the best run, inspect a scoreboard, summarize experiment outcomes, choose between candidates, explain why a strategy won or failed, debug failed or suspicious runs, or interpret QuantTradeAI metrics and artifacts with `quanttradeai runs list`, scoreboards, comparisons, and files under `runs/` or `reports/`.
---

# Run Analysis

Use this skill inside a user-created QuantTradeAI workspace after strategy, model, paper, live, or deployment workflows have already produced outputs. Treat the installed `quanttradeai` CLI and generated artifacts as the source of truth.

This skill is read and analysis oriented. Do not edit strategy configs by default. Do not promote, deploy, run paper, or run live unless the user explicitly asks to move to the next stage. If analysis suggests a winner, recommend promotion or the next experiment as a next step, but do not execute it automatically.

## Core Commands

Start with run discovery:

```bash
quanttradeai runs list
```

Use filters when the target family is known:

```bash
quanttradeai runs list --type research
quanttradeai runs list --type agent --mode backtest --status success
quanttradeai runs list --type agent --mode paper --status failed
```

Use scoreboards to rank candidates:

```bash
quanttradeai runs list --scoreboard
quanttradeai runs list --type agent --mode backtest --scoreboard --sort-by net_sharpe
quanttradeai runs list --type research --scoreboard
```

Use JSON when structured records are easier to inspect:

```bash
quanttradeai runs list --json
quanttradeai runs list --type batch --json
quanttradeai runs list --type agent --mode backtest --scoreboard --json
```

Use explicit comparison for 2-4 compatible finalists:

```bash
quanttradeai runs list --compare <run_a> --compare <run_b>
quanttradeai runs list --compare <run_a> --compare <run_b> --sort-by net_sharpe
quanttradeai runs list --compare <run_a> --compare <run_b> --json
```

Compare mode requires all runs to be from the same family: `research`, `agent/backtest`, `agent/paper`, or `agent/live`. Do not combine `--compare` with `--type`, `--mode`, `--status`, `--limit`, or `--scoreboard`.

## Artifact Reading Order

Terminal output is not enough for final conclusions. Always use durable artifacts before declaring a winner or explaining failure.

For a single run, inspect:

1. `summary.json`: status, run ID, name, timestamps, symbols, warnings, artifact paths.
2. `summary.json` -> `run_result` when present: compact outcome, ranked candidates, failures, or metric summary.
3. `metrics.json`: actual normalized metrics used by scoreboards.
4. `resolved_project_config.yaml`: exact config that ran.
5. `backtest_summary.json` when present: automatic model backtest details.

For a batch or sweep, inspect:

1. Batch `summary.json`.
2. Batch `scoreboard.json` for rankings.
3. Batch `results.json` to map ranked candidates to child run IDs, parameters, statuses, failures, and artifact paths.
4. Top child `summary.json`.
5. Top child `metrics.json`.
6. Top child `resolved_project_config.yaml`.

Use large line-delimited artifacts only when needed:

- `decisions.jsonl`: decision-level debugging for backtest, paper, or live behavior.
- `executions.jsonl`: execution-level debugging for paper/live fills, orders, or execution counts.
- `prompt_samples.json`: LLM or hybrid behavior debugging.
- `deployment_manifest.json`: deployment bundle analysis under `reports/deployments/...`.

Avoid reading huge JSONL files wholesale; sample the first/last lines or filter for relevant symbols, timestamps, actions, errors, or run IDs.

## Ranking Rules

For backtests, start with `net_sharpe` unless the user specifies another objective. Then check:

- `net_pnl`
- `net_mdd` or drawdown fields
- decision or trade count
- failures and warnings
- data window, costs, feature assumptions, and risk settings in `resolved_project_config.yaml`

For paper/live runs, focus more on:

- `total_pnl`
- portfolio value
- execution count
- decision count
- risk status
- failures and warnings

For model research, classification metrics alone are not enough when backtest metrics exist. Start from model metrics such as accuracy and F1, then verify backtest metrics, the configured test window, labels, feature set, and costs.

High rankings with sparse trades, missing metrics, failed children, narrow date windows, zero costs, or aggressive risk settings are suspicious. Mark these as caveats or say there is no clear winner.

## Practical Workflows

### Analyze The Latest Run

1. Run `quanttradeai runs list --limit 5`.
2. Identify the latest relevant run by type, mode, status, and name.
3. Open its `summary.json`, `metrics.json`, and `resolved_project_config.yaml`.
4. Report what ran, whether it succeeded, key metrics, warnings, and whether more evidence is needed.

### Analyze A Batch Or Sweep

1. Find batch records with `quanttradeai runs list --type batch --json`.
2. Open the batch `summary.json`.
3. Open `scoreboard.json` to see ranked records.
4. Open `results.json` to map the top rows to child runs and parameter variants.
5. Inspect the top child run artifacts before recommending anything.
6. Review failed child entries and include their error patterns.

### Compare Explicit Run IDs

1. Use `quanttradeai runs list --compare <run_a> --compare <run_b>`.
2. Confirm the comparison is same-family.
3. Review the metrics table.
4. Review config deltas from `resolved_project_config.yaml`.
5. Inspect each run's summary warnings and artifact paths.
6. Decide whether the metric gap is explained by meaningful config differences or by noise, missing data, or run quality issues.

### Explain Why The Top Run Won

1. Identify the ranking metric and objective.
2. Compare the winner against closest alternatives.
3. Read `results.json` for the winner's parameters if it came from a batch/sweep.
4. Read the winner's child `resolved_project_config.yaml`.
5. Verify the win is not only caused by sparse trades, no costs, a tiny test window, or failed competitors.

### Identify Failed Or Suspicious Runs

1. Run `quanttradeai runs list --status failed --json`.
2. Inspect failed `summary.json` files for `error`, `warnings`, and artifact pointers.
3. For failed batch children, use batch `results.json` and per-child logs if present.
4. Use JSONL files only when the failure requires decision/execution-level evidence.
5. Report root-cause evidence separately from speculation.

### Detect Insufficient Evidence

Say there is no clear winner when:

- top candidates have missing or invalid metrics
- all candidates failed or most child runs failed
- trade or decision count is too low to trust
- the winner only improves one metric while worsening drawdown, PnL, or risk
- configs differ in ways that make the comparison unfair
- model research has classification metrics but no useful backtest evidence

## Reporting Standards

When reporting back, include:

- Runs and artifacts inspected.
- Winner or `no clear winner`.
- Key metrics and the objective used for ranking.
- Closest alternatives.
- Config differences that mattered.
- Failures, warnings, missing metrics, or suspicious artifacts.
- Caveats about date windows, costs, risk settings, sparse trades, execution counts, model overfitting, or data assumptions.
- Recommended next action.

Keep conclusions proportional to the evidence. Recommend the next stage only after artifact-backed analysis supports it.

## More Detail

For product documentation, use:

- https://akkijoshi.gitbook.io/quanttradeai/
- https://github.com/AKKI0511/QuantTradeAI/tree/main/docs
