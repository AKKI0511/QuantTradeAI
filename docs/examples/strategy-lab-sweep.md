# Strategy Lab Sweep

## Goal

Use a coding agent to compare rule-based RSI mean-reversion and SMA crossover strategies on `AAPL` and `MSFT` using daily bars from `2020-01-01` through `2024-12-31`.

The agent should edit `config/project.yaml`, run backtest sweeps only, inspect artifacts, and recommend a candidate without implying that the result will be profitable out of sample. Results depend on the data source, date window, transaction costs, and risk settings.

## Best for

| Use this when | Avoid this when |
| --- | --- |
| You want a YAML-first rule strategy lab. | You need model training or promotion. |
| You want an agent to rank many deterministic variants. | You want paper or live trading in the same pass. |
| You care about artifact-backed selection instead of terminal output. | You do not have enough data coverage for the requested symbols/window. |

## Setup

```bash
quanttradeai init momentum-lab --template strategy-lab
cd momentum-lab
```

## Prompt to give your coding agent

```text
You are inside a QuantTradeAI workspace. Use `config/project.yaml` and the `quanttradeai` CLI.

Research RSI mean-reversion and SMA crossover strategies on AAPL and MSFT from 2020-01-01 to 2024-12-31 using daily data. Keep the work YAML-first. Adjust the strategy-lab config to include realistic parameter sweeps for RSI thresholds, SMA windows, and basic risk settings. Run backtests and sweeps only. Rank candidates by net Sharpe first, then check net PnL, max drawdown, trade count, and failures. Read `scoreboard.json`, `results.json`, and the top child run artifacts before recommending a winner. Do not run paper or live mode.
```

## What the agent should do

| Area | Expected YAML change |
| --- | --- |
| `data` | Set `symbols` to `AAPL` and `MSFT`, `timeframe` to `1d`, and the historical window to `2020-01-01` through `2024-12-31`. For a pure rule backtest, keep the backtest window aligned with the same dates unless the researcher asks for a separate holdout. |
| `features.definitions` | Keep `rsi_14`. Add SMA definitions that match the windows being tested, such as `sma_10`, `sma_20`, `sma_50`, and `sma_100`. |
| `agents` | Keep one `rule` agent for `rsi_threshold` and one `rule` agent for `sma_crossover`. Make sure each agent's `context.features` includes every feature it may use during a sweep. |
| `agents[].risk` | Add realistic scalar sizing values, for example `max_position_pct`, so sweeps can compare conservative versus more aggressive exposure. |
| `sweeps` | Use `agent_backtest` sweeps. RSI sweeps should vary `rule.buy_below`, `rule.sell_above`, and risk sizing. Keep the generated `sma_risk_grid` sweep name and expand its parameters so it varies `rule.fast_feature`, `rule.slow_feature`, and risk sizing. |

> [!NOTE]
> Agent sweeps mutate scalar leaves under the selected agent in generated child configs. They do not edit the source YAML and they do not change top-level feature generation settings unless the agent first defines the needed features.

## Commands it will likely run

```bash
quanttradeai validate -c config/project.yaml
quanttradeai agent run --sweep rsi_threshold_grid -c config/project.yaml --mode backtest --max-concurrency 4
quanttradeai agent run --sweep sma_risk_grid -c config/project.yaml --mode backtest --max-concurrency 4
quanttradeai runs list --type agent --mode backtest --scoreboard --sort-by net_sharpe
```

The exact parameters can differ, but each configured sweep should run with `--mode backtest`.

## Artifacts to inspect

| Artifact | Why it matters |
| --- | --- |
| `runs/agent/batches/.../scoreboard.json` | Ranked records, usually sorted by `net_sharpe` for backtest batches. |
| `runs/agent/batches/.../results.json` | Maps each ranked candidate to its child run ID, parameters, status, and failure details. |
| `runs/agent/batches/.../summary.json` | Batch status and compact `run_result` winner/failure analysis. |
| `runs/agent/backtest/.../summary.json` | Confirms the top child completed successfully and records the exact artifact paths. |
| `runs/agent/backtest/.../metrics.json` | Net Sharpe, net PnL, net max drawdown, and decision count for the child run. |
| `runs/agent/backtest/.../resolved_project_config.yaml` | The generated config that actually ran for the child candidate. |
| `runs/agent/backtest/.../ledger.csv` | Trade/fill activity when trades occurred. If absent, use `decision_count` and `decisions.jsonl` to understand inactivity. |

## Expected outcome

The expected result is a ranked set of RSI and SMA strategy variants under `runs/agent/batches/...`, with child backtest runs under `runs/agent/backtest/...`.

A useful recommendation should explain:

- The winning sweep and child run ID.
- The parameter values that produced the top result.
- Whether the result survived checks on net PnL, net max drawdown, trade activity, and failures.
- Any candidates rejected because they were inactive, failed validation, or only looked good on one metric.
- The artifact paths used to support the recommendation.

## Common mistakes

| Mistake | Why it matters |
| --- | --- |
| Recommending a winner from CLI stdout only. | The durable evidence is in `scoreboard.json`, `results.json`, and child run artifacts. |
| Running `--mode paper` or `--mode live`. | This workflow is backtest-only. |
| Sweeping a feature name that is not in `context.features`. | Rule validation or runtime feature resolution can fail. |
| Trying to agent-sweep top-level feature settings. | Agent sweeps only override scalar leaves on the selected agent. |
| Ranking by Sharpe without checking drawdown and trade activity. | A high Sharpe can still be fragile, inactive, or driven by a narrow date window. |
| Treating a backtest winner as deployable. | Backtests are research evidence, not approval for paper or live operation. |
