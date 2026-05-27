# Research Commands

`quanttradeai research run` runs the model research workflow defined in the project YAML. It validates the project config, compiles runtime model/feature/backtest configs, trains models, evaluates them, and records the run under `runs/`.

## When To Use

Use research runs when the project goal is to train and evaluate model artifacts from `research` config.

Use `quanttradeai agent run` instead when the thing you want to run is a YAML-defined trading agent from the `agents` list.

## Syntax

```bash
quanttradeai research run [OPTIONS]
```

```bash
quanttradeai research run
quanttradeai research run -c config/project.yaml
quanttradeai research run --sweep tuning-grid --max-concurrency 2
quanttradeai research run --skip-validation
```

## Options

| Option | Default | Required | Description |
|---|---:|:---:|---|
| `-c, --config TEXT` | `config/project.yaml` | No | Path to the project config YAML. |
| `--skip-validation` | `false` | No | Skip data-quality validation before training and automatic backtests. Project config validation still runs. |
| `--sweep TEXT` | `None` | No | Run every expanded variant for the named research sweep. The sweep must have `kind: research_run`. |
| `--max-concurrency INTEGER` | `1` | No | Maximum concurrent child runs when using `--sweep`. Must be at least `1`. |

## Single Run Vs Sweep

| Mode | Command shape | What runs |
|---|---|---|
| Single research run | `quanttradeai research run` | One workflow from the resolved project config. |
| Research sweep | `quanttradeai research run --sweep <name>` | One child research run per expanded sweep variant. |

Research sweeps expand scalar config overrides from `sweeps[]`, write variant outputs, then rank child runs using scoreboard metrics. If backtest metrics exist, research sweep ranking prefers `net_sharpe`; otherwise it uses `accuracy`.

## Config Sections Used

Research runs use:

- `project`
- `data`
- `features`
- `research`
- `sweeps` when `--sweep` is used

The runtime compiler converts these sections into focused runtime files for model training, feature generation, and backtesting. See [`docs/config/`](../config/) for the project YAML shape.

`research.enabled` must allow a normal research run. If it is disabled, `quanttradeai research run` fails during runtime config compilation.

## Reads

```text
config/project.yaml
```

The run also reads cached or fetched market data according to the compiled runtime model config.

## Writes / Artifacts

Single research runs write:

```text
runs/research/<timestamp>_<project_name>/
```

Important artifacts include:

| Artifact | Purpose |
|---|---|
| `resolved_project_config.yaml` | Resolved project config snapshot for the run. |
| `runtime_model_config.yaml` | Runtime model config compiled from project YAML. |
| `runtime_features_config.yaml` | Runtime feature config compiled from project YAML. |
| `runtime_backtest_config.yaml` | Runtime execution/backtest config compiled from project YAML. |
| `metrics.json` | Normalized research and automatic backtest metrics. |
| `summary.json` | Durable run record discovered by `quanttradeai runs list`. |
| `backtest_summary.json` | Automatic per-symbol model backtest summary when backtests run. |

The training pipeline writes model experiment artifacts under:

```text
models/experiments/<timestamp>/
```

Research sweeps write a batch directory under:

```text
runs/research/batches/<timestamp>_<project_name>_<sweep_name>/
```

Batch artifacts include:

- `resolved_project_config.yaml`
- `results.json`
- `scoreboard.json`
- `summary.json`
- child research runs under `runs/research/...`
- variant experiment directories under `models/experiments/...`

Artifact details are summarized in [`docs/artifacts.md`](../artifacts.md).

## CLI Result

On success, the command prints compact JSON. A single run includes fields such as:

```json
{
  "run_id": "research/...",
  "status": "success",
  "run_dir": "runs/research/...",
  "run_type": "research",
  "mode": "research",
  "name": "..."
}
```

A sweep prints a batch result with success/failure counts and a `winner` when a successful child can be ranked.

## Examples

Run the default research workflow:

```bash
quanttradeai research run
```

Run with a specific project file:

```bash
quanttradeai research run -c config/project.yaml
```

Run a research sweep:

```bash
quanttradeai research run --sweep cost-grid --max-concurrency 4
```

Inspect results:

```bash
quanttradeai runs list --type research --scoreboard
```

## Expected Outcome

A successful research run produces a durable run record, trained model artifacts, metrics, and an automatic compact JSON result. A failed run still writes a `summary.json` with `status: failed` and an error message.

## Common Mistakes

| Mistake | Why it fails or confuses results |
|---|---|
| Running research when `research.enabled` does not allow it | Runtime config compilation requires research to be enabled for research runs. |
| Confusing research promotion with agent promotion | Research promotion copies trained model artifacts into configured stable model paths; agent promotion changes agent/deployment modes. |
| Using research run for a YAML-defined trading agent | Trading agents live under `agents[]`; run them with `quanttradeai agent run`. |
| Using `--skip-validation` to skip project validation | This flag skips data-quality validation, not project config validation. |
| Defining a sweep with the wrong kind | `research run --sweep` requires a `research_run` sweep. |

## Related Docs

- [`docs/config/`](../config/)
- [`docs/artifacts.md`](../artifacts.md)
