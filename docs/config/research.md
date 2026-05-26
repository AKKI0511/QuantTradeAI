# Research

The `research` section controls model training, label generation, evaluation split behavior, research backtest costs, and model promotion targets.

## Used By

| Command | How it uses `research` |
|---|---|
| `quanttradeai validate` | Validates supported model settings and promotion targets. |
| `quanttradeai research run -c <path>` | Requires `research.enabled: true`, compiles runtime configs, trains models, runs model backtests, and writes run artifacts. |
| `quanttradeai research run --sweep <name>` | Runs expanded `research_run` sweep variants. |
| `quanttradeai promote --run research/<run> -c <path>` | Copies successful research model artifacts into `research.promotion.targets`. |
| `quanttradeai agent run` | Can reuse compiled data/features settings, but agent runs are separate from research runs. |

## Supported Fields

| Field | Supported values | Notes |
|---|---|---|
| `research.enabled` | boolean | Must be `true` for `quanttradeai research run`. |
| `research.labels.type` | `forward_return` | The only supported label type. |
| `research.labels.horizon` | integer | Forward return horizon. Defaults to `5`. |
| `research.labels.buy_threshold` | number | Positive threshold. Defaults to `0.01`. |
| `research.labels.sell_threshold` | number | Negative threshold. Defaults to `-0.01`. Runtime labeling uses the larger absolute threshold. |
| `research.model.kind` | `classifier` | The only supported model kind. |
| `research.model.family` | `voting` | The only supported model family. |
| `research.model.tuning.enabled` | boolean | Enables Optuna hyperparameter tuning. |
| `research.model.tuning.trials` | integer | Number of Optuna trials. Must be at least `1`. |
| `research.evaluation.split` | `time_aware` | The only supported split mode. |
| `research.evaluation.use_configured_test_window` | boolean | When true, runtime uses `data.test_start` and `data.test_end`. When false, the runtime clears them and uses chronological fallback splitting. |
| `research.backtest.costs.enabled` | boolean | Enables transaction cost injection into research model backtests. |
| `research.backtest.costs.bps` | number | Transaction cost in basis points. |
| `research.promotion.targets` | list | Stable model destinations for successful research promotion. |

Promotion target fields:

| Field | Rule |
|---|---|
| `name` | Must be non-empty and unique. |
| `symbol` | Must reference one of `data.symbols`. |
| `path` | Must be project-relative, resolve inside the project root, and live under `models/`. Paths must be unique. |

## Example

```yaml
research:
  enabled: true
  labels:
    type: forward_return
    horizon: 5
    buy_threshold: 0.01
    sell_threshold: -0.01
  model:
    kind: classifier
    family: voting
    tuning:
      enabled: true
      trials: 50
  evaluation:
    split: time_aware
    use_configured_test_window: true
  backtest:
    costs:
      enabled: true
      bps: 5
  promotion:
    targets:
      - name: aapl_daily_classifier
        symbol: AAPL
        path: models/promoted/aapl_daily_classifier
```

## How Research Run Uses It

`quanttradeai research run` performs these runtime steps:

1. Validate the project file.
2. Write `resolved_project_config.yaml`.
3. Compile `runtime_model_config.yaml`, `runtime_features_config.yaml`, and `runtime_backtest_config.yaml`.
4. Fetch historical data.
5. Generate features and labels.
6. Split train/test chronologically.
7. Train one voting classifier per symbol.
8. Save model artifacts under a research experiment directory.
9. Run model backtests for trained symbols.
10. Write run summaries and metrics.

Research promotion is intentionally separate. After a successful research run, `quanttradeai promote --run research/<run> -c <path>` copies trained symbol directories from the run's experiment directory into the configured `models/...` target paths and writes `promotion_manifest.json` inside each promoted model directory.

## Expected Outcomes

Research run directories are written under `runs/research/`. Typical artifacts include:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `runtime_backtest_config.yaml`
- `summary.json`
- `metrics.json`
- `backtest_summary.json` when automatic model backtests produce payloads
- an experiment directory under `models/experiments/...`
- experiment files such as `results.json`, `test_window_coverage.json`, and `preprocessing.json`

Promotion writes:

- promoted model directories under `research.promotion.targets[].path`
- `promotion_manifest.json` in each promoted target directory

## Common Mistakes

| Mistake | Result |
|---|---|
| Running `research run` with `research.enabled: false` | The command fails before training. |
| Leaving `research.promotion.targets` empty and then promoting a research run | Promotion fails because there is no stable destination. |
| Pointing a promotion target outside `models/` | Validation or promotion fails. |
| Confusing research runs with agent runs | Research trains model artifacts; agent runs test or operate a YAML-defined trading agent. |
| Assuming paper or live agent modes are enabled by a research run | Agent modes are controlled by `agents[].mode` and promotion commands. |

