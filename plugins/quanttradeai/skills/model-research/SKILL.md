---
name: model-research
description: Practical cookbook for using QuantTradeAI inside an installed user workspace to train and evaluate ML research models, build classifiers, run research sweeps, inspect research artifacts, promote successful research runs into stable model paths, and use promoted artifacts in model or hybrid agent backtests. Use when a user asks to train a model, run ML research, build a classifier, evaluate model performance, promote a research run, create or use a model agent, connect research output to a deployable agent, or run research sweeps through `quanttradeai` and project YAML.
---

# Model Research

Use this skill inside a user-created QuantTradeAI workspace, usually initialized with `quanttradeai init <workspace>`. Treat the installed `quanttradeai` CLI, the workspace YAML, and the generated artifacts as the source of truth. This skill is for operating a QuantTradeAI workspace, not for contributing to the QuantTradeAI package itself.

Stay in research and backtest mode by default. Do not run paper/live mode, deploy, use broker credentials, or claim profitability from classification metrics alone unless the user explicitly asks for a later stage.

## Research Vs Agent Runs

Use `quanttradeai research run` when the goal is to train and evaluate model artifacts from the `research` section. A research run performs data loading, feature generation, labels, chronological train/test evaluation, model training, automatic model backtests when available, and artifact recording.

Use `quanttradeai agent run` only after a promoted model artifact exists and a YAML-defined `kind: model` or `kind: hybrid` trading agent should be backtested from that stable path. Agent runs do not train models.

## YAML Areas To Inspect

Start with the user-provided config path; otherwise use `config/project.yaml`.

Inspect and edit only what the research question requires:

- `data`: symbols, timeframe, historical date range, test window, cache settings.
- `features`: definitions used for training and later model-agent feature consistency.
- `research.enabled`: must be `true` for `research run`.
- `research.labels`: `type: forward_return`, horizon, buy/sell thresholds.
- `research.model`: `kind: classifier`, `family: voting`, tuning enabled/trials.
- `research.evaluation`: `split: time_aware`, usually `use_configured_test_window: true`.
- `research.backtest`: transaction cost settings for automatic model backtests.
- `research.promotion.targets`: stable project-relative destinations under `models/`.
- `sweeps`: `kind: research_run` grids for research parameters.
- `agents`: only when wiring an existing promoted model into a `model` or `hybrid` agent.

Keep the workflow YAML-first. Avoid ad hoc training scripts.

## Validate Before Training

Validation confirms the project config can be consumed and catches invalid research settings, promotion targets, unsupported sweeps, missing model/hybrid paths, and other config problems.

```bash
quanttradeai validate -c <config>
```

Fix validation errors before training. For promotion targets, make sure each target has a unique `name`, a `symbol` present in `data.symbols`, and a project-relative `path` under `models/`.

## Run One Research Workflow

Use a single research run to train and evaluate the configured model workflow.

```bash
quanttradeai research run -c <config>
```

Expected run path:

```text
runs/research/<timestamp>_<project_name>/
```

Training artifacts are also written under:

```text
models/experiments/<timestamp>/
```

Inspect before promotion:

- `summary.json`: status, run ID, warnings, artifact paths, experiment directory.
- `summary.json.run_result`: compact outcome when present.
- `metrics.json`: research metrics by symbol and automatic backtest metrics when available.
- `backtest_summary.json`: model backtest details when produced.
- `resolved_project_config.yaml`: exact data, features, labels, evaluation, costs, and promotion targets that ran.

Do not judge a model from accuracy/F1 alone. Check the test window, class balance if visible in artifacts, backtest metrics, warnings, and whether the configured test period is truly out of sample for the question.

## Run A Research Sweep

Use a research sweep to compare training/evaluation parameter variants. The sweep must be `kind: research_run`.

```bash
quanttradeai research run -c <config> --sweep <sweep_name> --max-concurrency <n>
```

Example shape:

```yaml
sweeps:
  - name: label_cost_grid
    kind: research_run
    parameters:
      - path: research.labels.horizon
        values: [3, 5, 10]
      - path: research.backtest.costs.bps
        values: [1, 5]
      - path: research.model.tuning.trials
        values: [20, 50]
```

Supported research sweep paths include selected scalar `data` leaves, existing scalar leaves under `research.labels`, `research.backtest.costs`, `research.model.tuning`, `research.evaluation.use_configured_test_window`, and `features.<feature_name>.params.<param>`.

Research sweeps do not mutate the source YAML. They generate variant configs for child runs and batch artifacts.

Expected batch path:

```text
runs/research/batches/<timestamp>_<project_name>_<sweep_name>/
```

For sweeps, inspect:

- Batch `summary.json`: status, counts, warnings, `run_result`.
- Batch `scoreboard.json`: ranked child records. Ranking prefers `net_sharpe` when backtest metrics exist; otherwise it uses accuracy.
- Batch `results.json`: child run IDs, parameters, statuses, warnings, errors, and artifact paths.
- Top child `summary.json`, `metrics.json`, `backtest_summary.json` when present, and `resolved_project_config.yaml`.

## Rank And Compare Research Runs

Use the research scoreboard to find candidates:

```bash
quanttradeai runs list --type research --scoreboard
```

Use explicit comparison for 2-4 research finalists:

```bash
quanttradeai runs list --compare research/<run_a> --compare research/<run_b>
```

Before selecting a run for promotion, prefer candidates that are successful, have coherent out-of-sample/test-window behavior, have acceptable backtest metrics when available, and do not depend on a fragile or accidental configuration.

## Promote A Research Run

Promote only after inspecting artifacts and choosing a successful research run.

```bash
quanttradeai promote --run research/<run_id> -c <config>
```

Research promotion copies trained symbol directories from the run's experiment directory into the configured `research.promotion.targets[].path` and writes `promotion_manifest.json` inside each promoted target. It stabilizes model paths for later agent usage; it does not by itself prove the model is ready for paper/live mode.

After promotion, verify:

- Each target directory exists under `models/...`.
- Each target has `promotion_manifest.json`.
- The manifest source run ID and symbol match the selected research run.
- Any model or hybrid agent that will use the artifact points to the promoted path, not a timestamped `models/experiments/...` path.

## Backtest A Model Or Hybrid Agent

Only use `agents` after promotion or when a valid promoted artifact already exists.

For a model agent:

```yaml
agents:
  - name: paper_momentum
    kind: model
    mode: paper
    execution:
      backend: simulated
    model:
      path: models/promoted/aapl_daily_classifier
    risk:
      max_position_pct: 0.05
```

For a hybrid agent, `model_signal_sources[].path` should point at a promoted model path and `context.model_signals` should reference the source name.

Validate after adding or changing the agent:

```bash
quanttradeai validate -c <config>
```

Then run a backtest:

```bash
quanttradeai agent run --agent <model_agent_name> -c <config> --mode backtest
```

Inspect the model-agent backtest artifacts under `runs/agent/backtest/...`, especially `summary.json`, `metrics.json`, and `resolved_project_config.yaml`, to confirm the promoted model path actually ran.

## Reporting Standards

When reporting back, include:

- Research config changes made.
- Commands run.
- Validation and training status.
- Artifact paths inspected.
- Key model metrics by symbol.
- Backtest summary and backtest metrics when available.
- Whether promotion was performed.
- Promoted model path and `promotion_manifest.json` path if created.
- Model-agent or hybrid-agent backtest result if run.
- Caveats about overfitting, date windows, features, data assumptions, classification metrics, and backtest limits.
- Recommended next research step.

Keep recommendations evidence-based. A successful research run or high classifier score is not enough to claim a profitable strategy.

## More Detail

For product documentation, use:

- https://akkijoshi.gitbook.io/quanttradeai/
- https://github.com/AKKI0511/QuantTradeAI/tree/main/docs
