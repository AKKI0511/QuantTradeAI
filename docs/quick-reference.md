# Quick Reference

Common commands, patterns, and examples for QuantTradeAI.

Project-defined `agent run --mode paper` defaults to deterministic replay through `data.streaming.replay`. Generated deployment bundles stay on the real-time paper path.

## CLI Commands

```bash
# Show help
poetry run quanttradeai --help

# Initialize and validate canonical project config
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml

# Run canonical research workflow
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai promote --run research/<run_id> -c config/project.yaml
poetry run quanttradeai runs list
poetry run quanttradeai runs list --scoreboard --sort-by net_sharpe

# Run a YAML-defined llm or hybrid agent
poetry run quanttradeai init --template llm-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
# Promote the successful backtest run before starting paper mode
poetry run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
# Local paper mode replays historical OHLCV by default
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode paper

# Backtest every configured project agent together
poetry run quanttradeai agent run --all -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --all -c config/project.yaml --mode backtest --max-concurrency 4
# Replay-backed paper batch across every configured project agent
poetry run quanttradeai agent run --all -c config/project.yaml --mode paper
poetry run quanttradeai agent run --all -c config/project.yaml --mode paper --max-concurrency 4

# Backtest one sweep of agent parameter variants
poetry run quanttradeai agent run --sweep rsi_threshold_grid -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --sweep rsi_threshold_grid -c config/project.yaml --mode backtest --max-concurrency 4

# Generate a Docker Compose bundle for the paper agent
# Generated bundles disable replay and expect real-time streaming settings
poetry run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target docker-compose

# Import an existing legacy config/ directory into the canonical workflow
poetry run quanttradeai validate --legacy-config-dir config
poetry run quanttradeai research run --legacy-config-dir config

# Legacy compatibility commands
poetry run quanttradeai fetch-data
poetry run quanttradeai fetch-data --refresh

# Train models
poetry run quanttradeai train
poetry run quanttradeai train --skip-validation  # bypass data-quality gate

# Evaluate model
poetry run quanttradeai evaluate -m models/experiments/<timestamp>/<SYMBOL>

# Backtest a saved model (end-to-end)
poetry run quanttradeai backtest-model -m models/experiments/<timestamp>/<SYMBOL> \
  -c config/model_config.yaml -b config/backtest_config.yaml \
  --cost-bps 5 --slippage-fixed 0.01 --liquidity-max-participation 0.25 \
  --skip-validation  # optional

# Live trading (streaming)
# This path still uses model_config.yaml + features_config.yaml + streaming/risk/position_manager YAMLs
poetry run quanttradeai live-trade -m models/experiments/<timestamp>/<SYMBOL> \
  -c config/model_config.yaml \
  -s config/streaming.yaml \
  --risk-config config/risk_config.yaml \
  --position-manager-config config/position_manager.yaml \
  --initial-capital 1000000 --min-history 220 --history-window 512
```

Canonical research artifacts:
- `runs/research/<timestamp>_<project>/resolved_project_config.yaml`
- `runs/research/<timestamp>_<project>/runtime_model_config.yaml`
- `runs/research/<timestamp>_<project>/runtime_features_config.yaml`
- `runs/research/<timestamp>_<project>/runtime_backtest_config.yaml`
- `runs/research/<timestamp>_<project>/summary.json`
- `runs/research/<timestamp>_<project>/metrics.json`
- `runs/research/<timestamp>_<project>/backtest_summary.json`

Canonical agent backtest artifacts:
- `runs/agent/backtest/<timestamp>_<agent>/resolved_project_config.yaml`
- `runs/agent/backtest/<timestamp>_<agent>/runtime_model_config.yaml`
- `runs/agent/backtest/<timestamp>_<agent>/runtime_features_config.yaml`
- `runs/agent/backtest/<timestamp>_<agent>/summary.json`
- `runs/agent/backtest/<timestamp>_<agent>/metrics.json`
- `runs/agent/backtest/<timestamp>_<agent>/decisions.jsonl`

Canonical agent paper artifacts:
- `runs/agent/paper/<timestamp>_<agent>/resolved_project_config.yaml`
- `runs/agent/paper/<timestamp>_<agent>/runtime_model_config.yaml`
- `runs/agent/paper/<timestamp>_<agent>/runtime_features_config.yaml`
- `runs/agent/paper/<timestamp>_<agent>/runtime_streaming_config.yaml`
- `runs/agent/paper/<timestamp>_<agent>/summary.json`
- `runs/agent/paper/<timestamp>_<agent>/metrics.json`
- `runs/agent/paper/<timestamp>_<agent>/executions.jsonl`
- `runs/agent/paper/<timestamp>_<agent>/decisions.jsonl` for `rule`, `llm`, and `hybrid`
- `runs/agent/paper/<timestamp>_<agent>/prompt_samples.json` for `llm` and `hybrid`
- `runs/agent/paper/<timestamp>_<agent>/replay_manifest.json` when replay is enabled

Canonical deployment bundle artifacts:
- `reports/deployments/<agent>/<timestamp>/docker-compose.yml`
- `reports/deployments/<agent>/<timestamp>/Dockerfile`
- `reports/deployments/<agent>/<timestamp>/.env.example`
- `reports/deployments/<agent>/<timestamp>/README.md`
- `reports/deployments/<agent>/<timestamp>/resolved_project_config.yaml`
- `reports/deployments/<agent>/<timestamp>/deployment_manifest.json`

Canonical multi-agent batch artifacts:
- `runs/agent/batches/<timestamp>_<project>_backtest/batch_manifest.json`
- `runs/agent/batches/<timestamp>_<project>_backtest/results.json`
- `runs/agent/batches/<timestamp>_<project>_backtest/scoreboard.json`
- `runs/agent/batches/<timestamp>_<project>_backtest/scoreboard.txt`
- `runs/agent/batches/<timestamp>_<project>_paper/batch_manifest.json`
- `runs/agent/batches/<timestamp>_<project>_paper/results.json`
- `runs/agent/batches/<timestamp>_<project>_paper/scoreboard.json`
- `runs/agent/batches/<timestamp>_<project>_paper/scoreboard.txt`

Canonical sweep batch artifacts:
- `runs/agent/batches/<timestamp>_<project>_<sweep>_backtest/batch_manifest.json`
- `runs/agent/batches/<timestamp>_<project>_<sweep>_backtest/results.json`
- `runs/agent/batches/<timestamp>_<project>_<sweep>_backtest/scoreboard.json`
- `runs/agent/batches/<timestamp>_<project>_<sweep>_backtest/scoreboard.txt`

## Python API Patterns

### Data Loading
```python
from quanttradeai import DataLoader

# Initialize and fetch data
loader = DataLoader("config/model_config.yaml")
data = loader.fetch_data(symbols=["AAPL", "TSLA"], refresh=True)

# Validate data and inspect per-symbol report
is_valid, report = loader.validate_data(data)
if not is_valid:
    print(report)
    # CLI commands write this report to models/experiments/<timestamp>/validation.json
    # or reports/backtests/<timestamp>/validation.json
    # canonical research runs also persist resolved configs under runs/research/<timestamp>_<project>/
```

### Feature Engineering
```python
from quanttradeai import DataProcessor

# Process raw data
processor = DataProcessor("config/features_config.yaml")
df_processed = processor.process_data(raw_df)

# Generate labels
df_labeled = processor.generate_labels(df_processed, forward_returns=5, threshold=0.01)
```

#### Multi-timeframe Features
Configure derived intraday signals directly in `config/features_config.yaml`:

```yaml
multi_timeframe_features:
  enabled: true
  operations:
    - type: ratio          # Options: ratio, delta, pct_change, rolling_divergence
      timeframe: "1h"      # Matches secondary_timeframes in model config
      base: close          # open, high, low, close, or volume
      stat: last           # Optional override; defaults to loader aggregate
    - type: rolling_divergence
      timeframe: "30m"
      base: volume
      stat: sum
      rolling_window: 5    # Required for rolling_divergence
```

When enabled, the processor emits columns such as `mtf_ratio_close_1h_last`
after the technical indicator step, making secondary timeframe aggregates
immediately available to the model pipeline.

### Model Training
```python
from quanttradeai import MomentumClassifier

# Initialize and train
classifier = MomentumClassifier("config/model_config.yaml")
X, y = classifier.prepare_data(df_labeled)
classifier.train(X, y)

# Save model
classifier.save_model("models/promoted/aapl_daily_classifier")
```

### Backtesting
```python
from quanttradeai import simulate_trades, compute_metrics

# Simulate trades with intrabar fills and market impact
df_trades = simulate_trades(
    df_labeled,
    execution={
        "impact": {
            "enabled": True,
            "model": "linear",
            "alpha": 0.5,
            "beta": 0.0,
            "average_daily_volume": 1_000_000,
        },
        "intrabar": {"enabled": True, "synthetic_ticks": 20, "volatility": 0.01},
        "borrow_fee": {"enabled": True, "rate_bps": 100},
    },
)

# Calculate metrics
metrics = compute_metrics(df_trades, risk_free_rate=0.02)
```

### Backtest a Saved Model
```bash
# Uses saved feature_columns and execution config to produce PnL metrics
poetry run quanttradeai backtest-model -m models/experiments/<timestamp>/<SYMBOL> \
  -c config/model_config.yaml -b config/backtest_config.yaml

# Artifacts are saved under:
# reports/backtests/<run_timestamp>/<SYMBOL>/{metrics.json,equity_curve.csv,ledger.csv}
# reports/backtests/<run_timestamp>/portfolio/{metrics.json,equity_curve.csv}
```

## Technical Indicators

```python
from quanttradeai.features import technical as ta

# Moving averages
sma_20 = ta.sma(df["Close"], 20)
ema_20 = ta.ema(df["Close"], 20)

# Momentum indicators
rsi_14 = ta.rsi(df["Close"], 14)
macd_df = ta.macd(df["Close"])
stoch_df = ta.stochastic(df["High"], df["Low"], df["Close"])
```

## Risk Management

```python
from quanttradeai import apply_stop_loss_take_profit, position_size

# Apply risk rules
df_with_risk = apply_stop_loss_take_profit(df, stop_loss_pct=0.02, take_profit_pct=0.04)

# Calculate position size
qty = position_size(capital=10000, risk_per_trade=0.02, stop_loss_pct=0.05, price=150.0)
```

```python
from quanttradeai.trading import DrawdownGuard, PortfolioManager

# Allocate capital across multiple symbols with drawdown protection.
# Passing drawdown_guard wires it through an internal RiskManager.
guard = DrawdownGuard(config_path="config/risk_config.yaml")
pm = PortfolioManager(10000, drawdown_guard=guard)
pm.open_position("AAPL", price=150, stop_loss_pct=0.05)
pm.open_position("TSLA", price=250, stop_loss_pct=0.05)
print(f"Portfolio exposure: {pm.risk_exposure:.2%}")
```

## Performance Metrics

```python
from quanttradeai.utils.metrics import classification_metrics, sharpe_ratio, max_drawdown

# Classification metrics
metrics = classification_metrics(y_true, y_pred)

# Trading metrics
sharpe = sharpe_ratio(returns, risk_free_rate=0.02)
mdd = max_drawdown(equity_curve)
```

## Visualization

```python
from quanttradeai.utils.visualization import plot_price, plot_performance

# Plot charts
plot_price(df, title="AAPL Price Chart")
plot_performance(equity_curve, title="Strategy Performance")
```

## Configuration

Use these pages instead of copying large config blocks out of the quick reference:

- [Configuration Overview](configuration.md)
- [Project Config (`project.yaml`)](configuration/project-yaml.md)
- [Runtime and Live Trading Configs](configuration/live-runtime-files.md)
- [Legacy Config Compatibility](configuration/legacy-configs.md)

Quick decision rule:

- Use `config/project.yaml` for `validate`, `research run`, `agent run`, and `agent run --sweep`
- Use `data.streaming.replay` for deterministic local paper runs; keep provider/websocket fields in place for later deployment or live promotion
- Use `quanttradeai runs list --scoreboard` to compare local research and agent runs by metrics
- Use the runtime YAML files for `live-trade`, `backtest-model`, and operational streaming setup

## Time-Aware Splitting

- In the canonical workflow, train/test splits respect `data.test_start` and `data.test_end` from `config/project.yaml`.
- `research.evaluation.use_configured_test_window: false` disables that explicit window and falls back to a chronological split.
- If only `test_start` is provided: train = dates < `test_start`; test = dates >= `test_start`.
- If neither is provided: a chronological split uses the last `training.test_size` fraction as test (default 0.2).

Hyperparameter tuning uses `TimeSeriesSplit(n_splits=training.cv_folds)` to avoid future leakage during CV.

## Streaming

```python
from quanttradeai.streaming.providers import (
    ProviderConfigValidator,
    ProviderDiscovery,
    ProviderHealthMonitor,
)

discovery = ProviderDiscovery()           # auto-discovers adapters with hot reload support
registry = discovery.discover()
adapter = registry.create_instance("example")

validator = ProviderConfigValidator()
model = validator.load_from_path("config/providers/example.yaml", environment="dev")
runtime = validator.validate(adapter, model)

monitor = ProviderHealthMonitor()
monitor.register_provider(adapter.provider_name, status_provider=adapter.get_health_status)

# inside an async context
await monitor.execute_with_health(adapter.provider_name, adapter.connect)
await monitor.execute_with_health(
    adapter.provider_name,
    lambda: adapter.subscribe(["AAPL"]),
)
# Use `adapter` with the legacy StreamingGateway or custom event loop as needed
```

Provider configuration file (`config/providers/example.yaml`):

```yaml
provider: example
environment: dev
environments:
  dev:
    asset_types: ["stocks", "crypto"]
    data_types: ["trades", "quotes"]
    options:
      mode: "realtime"
```

- Use `discovery.refresh()` to hot reload newly added adapters.
- `monitor.execute_with_health()` wraps connect/subscribe calls with circuit breaking and failover handling.
- Legacy `config/streaming.yaml` continues to control shared buffers, subscriptions, and rate limits for the built-in gateway.

### Position Manager

```python
from quanttradeai.streaming import StreamingGateway
from quanttradeai.trading import PositionManager

gw = StreamingGateway("config/streaming.yaml")
pm = PositionManager.from_config("config/position_manager.yaml")
pm.bind_gateway(gw, ["AAPL", "MSFT"])
```

The built-in `live-trade` command wires the position manager for you when `--position-manager-config` is provided.

## Error Handling

```python
try:
    # Fetch data
    data = loader.fetch_data()

    # Process data
    df_processed = processor.process_data(data["AAPL"])

    # Train model
    classifier.train(X, y)

except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Troubleshooting

### Data Issues
```python
# Check cache directory
import os
print(os.path.exists("data/raw"))

# Force refresh
data = loader.fetch_data(refresh=True)

# Check for NaN values
print(df.isnull().sum())
df = df.fillna(method="ffill")
```

### Model Issues
```python
# Check data shapes
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Check class distribution
print(pd.Series(y).value_counts())
```

### Backtesting Issues
```python
# Check label distribution
print(df["label"].value_counts())

# Ensure proper date index
print(df.index.dtype)
```

## Related Documentation

- **[Getting Started](getting-started.md)** - Installation and first steps
- **[API Reference](api/)** - Complete API documentation
- **[Configuration](configuration.md)** - Configuration guide
