# Research To Model Agent

## Goal

Use a coding agent to train a momentum-style classifier on `AAPL` and `MSFT` daily data from `2018-01-01` through `2024-12-31`, evaluate it on `2024-09-01` through `2024-12-31`, promote successful model artifacts, and run a model-agent backtest from a stable promoted path.

This is a research-to-agent handoff. It should prove what trained, what was promoted, and what the model agent backtested. It should not imply guaranteed profitability. Results depend on data quality, feature definitions, the test window, costs, and risk settings.

## Best for

| Use this when | Avoid this when |
| --- | --- |
| You want model training before agent backtesting. | You only need rule-based strategy sweeps. |
| You want stable promoted model paths instead of timestamped experiment paths. | You want paper or live execution in the same pass. |
| You want a coding agent to verify artifacts before recommending next steps. | You have not decided which promoted symbol model the model agent should use first. |

## Setup

```bash
quanttradeai init model-research-lab --template research
cd model-research-lab
```

## Prompt to give your coding agent

```text
You are inside a QuantTradeAI workspace. Use `config/project.yaml` and the `quanttradeai` CLI.

Train a momentum-style classifier for AAPL and MSFT using daily data from 2018-01-01 to 2024-12-31, with 2024-09-01 to 2024-12-31 as the test window. Use RSI and SMA-style technical features where supported by the current config schema. Keep the workflow YAML-first. Run the research workflow, inspect `summary.json`, `metrics.json`, `backtest_summary.json`, and `resolved_project_config.yaml`, then promote the best successful research run into a stable model path. After promotion, prepare or switch to a model-agent configuration that uses the promoted model and run a backtest. Do not run paper or live mode.
```

## What the agent should do

| Config section | What matters |
| --- | --- |
| `data` | Use `symbols: [AAPL, MSFT]`, `timeframe: 1d`, `start_date: "2018-01-01"`, `end_date: "2024-12-31"`, `test_start: "2024-09-01"`, and `test_end: "2024-12-31"`. |
| `features` | Keep RSI enabled and add SMA-style technical definitions that the current schema supports, such as `sma_20`, `sma_50`, and `sma_100`. |
| `research` | Keep `enabled: true`, use `classifier` and `voting`, keep `evaluation.use_configured_test_window: true`, and include realistic research backtest costs. |
| `research.promotion` | Define stable targets under `models/promoted/...`, usually one target per trained symbol, such as `aapl_daily_classifier` and `msft_daily_classifier`. |
| `agents` or model-agent config | After promotion, add or switch to a `kind: model` agent whose `model.path` points at a promoted target that exists. Align the backtest data symbols with the chosen promoted model path for the first model-agent run. |

> [!TIP]
> Promotion copies trained artifacts out of `models/experiments/...` into stable paths such as `models/promoted/aapl_daily_classifier`. The model agent should reference the stable promoted path, not a timestamped experiment directory.

## Commands it will likely run

```bash
quanttradeai validate -c config/project.yaml
quanttradeai research run -c config/project.yaml
quanttradeai runs list --type research --scoreboard --sort-by net_sharpe
quanttradeai promote --run research/<run_id> -c config/project.yaml
quanttradeai validate -c config/project.yaml
quanttradeai agent run --agent aapl_momentum_model -c config/project.yaml --mode backtest
```

The model-agent name can differ. The key requirement is that `agents[].model.path` points to an existing promoted artifact before the agent backtest runs.

## Artifacts to inspect

| Artifact | Why it matters |
| --- | --- |
| `runs/research/.../summary.json` | Confirms the research run status, run ID, experiment directory, and artifact paths. |
| `runs/research/.../metrics.json` | Shows research and automatic backtest metrics by symbol. |
| `runs/research/.../backtest_summary.json` | Captures the automatic model backtest summary when produced. |
| `runs/research/.../resolved_project_config.yaml` | Proves the data window, features, evaluation split, costs, and promotion targets used. |
| `models/experiments/.../<SYMBOL>/` | Source trained model artifacts created by the research run. |
| `models/promoted/.../promotion_manifest.json` | Proves which research run and symbol produced the stable promoted model. |
| `runs/agent/backtest/.../summary.json` | Confirms the model agent ran successfully and records the model path used. |
| `runs/agent/backtest/.../metrics.json` | Evaluates the model-agent backtest with net metrics. |
| `runs/agent/backtest/.../resolved_project_config.yaml` | Confirms the model-agent config used the promoted path, not the experiment path. |

## Expected outcome

The expected result is:

- A successful research run under `runs/research/...`.
- One or more promoted model directories under `models/promoted/...`.
- A `promotion_manifest.json` in each promoted model directory.
- A model-agent backtest under `runs/agent/backtest/...` that references a promoted model path.

A good final agent report should include the selected research run ID, promotion target path, promotion manifest path, model-agent run ID, the key metrics inspected, and any limitations found in the artifacts.

## Common mistakes

| Mistake | Why it matters |
| --- | --- |
| Adding a model agent before the promoted model path exists, then validating. | Validation can fail because `agents[].model.path` must resolve to an existing artifact. |
| Pointing the model agent at `models/experiments/...`. | Timestamped experiment paths are not stable handoff paths. |
| Promoting without checking `summary.json` and `metrics.json`. | Failed or partial runs should not be promoted. |
| Training on `AAPL` and `MSFT` but using one symbol's promoted classifier across both without calling it out. | The model-agent path is singular. For cleaner attribution, run separate model-agent configs per promoted symbol. |
| Ignoring `resolved_project_config.yaml`. | The source YAML can differ from the exact resolved config that executed. |
| Running paper or live mode. | This example stops at model-agent backtesting. |
