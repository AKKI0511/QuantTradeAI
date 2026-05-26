# Utility Commands

These lower-level commands operate on focused config files and artifacts. Use them when you need direct data fetching, direct model evaluation, or a simple CSV-driven backtest rather than the project-level research or agent workflows.

## `quanttradeai fetch-data`

Fetches OHLCV data using a model config and writes cache files.

### When To Use

Use it to warm or refresh the market-data cache before running model workflows.

### Syntax

```bash
quanttradeai fetch-data [OPTIONS]
```

```bash
quanttradeai fetch-data
quanttradeai fetch-data -c config/model_config.yaml
quanttradeai fetch-data --refresh
```

### Options

| Option | Default | Required | Description |
|---|---:|:---:|---|
| `-c, --config TEXT` | `config/model_config.yaml` | No | Path to the model/data config file. |
| `--refresh` | `false` | No | Force refresh of cached data. |

### Inputs

The command reads the model config's `data` section, including symbols, dates, timeframe, cache path, cache settings, and optional news settings.

### Outputs

It writes cached parquet data under the configured cache directory. The loader can write:

```text
<cache_dir>/<SYMBOL>_<timeframe>_data.parquet
<cache_dir>/<SYMBOL>_news.parquet
<cache_dir>/<SYMBOL>_data.parquet
```

The exact files depend on cache settings, timeframe settings, and whether news is enabled.

### Expected Outcome

Data is fetched or loaded from cache, then saved to disk. Failures are logged by symbol.

## `quanttradeai evaluate`

Evaluates a saved model directory against the dataset defined by a model config.

### When To Use

Use it when you already have a trained model directory and want fresh evaluation metrics without launching a full project-level research run.

### Syntax

```bash
quanttradeai evaluate --model-path MODEL_DIR [OPTIONS]
```

```bash
quanttradeai evaluate -m models/experiments/20260525_120000/AAPL
quanttradeai evaluate --model-path models/promoted/aapl_daily_classifier -c config/model_config.yaml
quanttradeai evaluate -m models/promoted/aapl_daily_classifier --skip-validation
```

### Options

| Option | Default | Required | Description |
|---|---:|:---:|---|
| `-m, --model-path TEXT` | none | Yes | Saved model directory. |
| `-c, --config TEXT` | `config/model_config.yaml` | No | Path to the model/data config file. |
| `--skip-validation` | `false` | No | Skip data-quality validation before evaluation. |

### Inputs

The command reads:

- the model config
- market data from cache or provider
- the saved model directory
- `feature_preprocessor.joblib` from the model directory when present
- `config/features_config.yaml` through the default `DataProcessor`; this command has no option for a separate features config path

### Outputs

It writes:

```text
<model-path>/validation.json
<model-path>/validation.csv
<model-path>/evaluation.json
```

Validation artifacts are skipped only when `--skip-validation` is set.

### Expected Outcome

The command logs per-symbol metrics and writes `evaluation.json` inside the model directory.

## `quanttradeai backtest`

Runs a CSV-driven backtest using a backtest config file and optional execution-cost overrides.

### When To Use

Use it for a focused backtest of a CSV that already contains the required price and signal columns. The CSV must include data accepted by `simulate_trades()`, including `Close` and `label`.

### Syntax

```bash
quanttradeai backtest [OPTIONS]
```

```bash
quanttradeai backtest
quanttradeai backtest -c config/backtest_config.yaml
quanttradeai backtest --cost-bps 5 --slippage-bps 2
quanttradeai backtest --liquidity-max-participation 0.1
```

### Options

| Option | Default | Required | Description |
|---|---:|:---:|---|
| `-c, --config TEXT` | `config/backtest_config.yaml` | No | Backtest config file. |
| `--cost-bps FLOAT` | `None` | No | Override transaction cost using basis points. |
| `--cost-fixed FLOAT` | `None` | No | Override transaction cost using a fixed amount. |
| `--slippage-bps FLOAT` | `None` | No | Override slippage using basis points. |
| `--slippage-fixed FLOAT` | `None` | No | Override slippage using a fixed amount. |
| `--liquidity-max-participation FLOAT` | `None` | No | Override liquidity max participation. |

### Inputs

The command reads:

```text
config/backtest_config.yaml
```

The config must include `data_path`, and the referenced CSV is loaded with `pandas.read_csv()`.

### Outputs

The command prints computed metrics as formatted JSON. It does not create a run record under `runs/`.

### Expected Outcome

The CLI loads the CSV, applies execution settings from the config plus any CLI overrides, runs `simulate_trades()`, computes metrics, and prints them.

## Common Mistakes

| Mistake | Why it matters |
|---|---|
| Expecting utility commands to write `runs/` records | These commands are direct helpers; use `research run` or `agent run` for durable run records. |
| Using `fetch-data` with `config/project.yaml` | `fetch-data` expects the focused model/data config shape. |
| Evaluating a missing or incomplete model directory | The model loader requires saved model files in `--model-path`. |
| Backtesting a CSV without a `label` column | `simulate_trades()` expects trading labels to drive positions. |
| Expecting `backtest` to save metrics artifacts | It prints JSON metrics to stdout only. |

## Related Docs

- [`docs/config/`](../config/)
- [`docs/artifacts.md`](../artifacts.md)
