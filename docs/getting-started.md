# Getting Started

Welcome to QuantTradeAI! This guide will help you get up and running quickly.

## 🚀 Quick Installation

### Prerequisites
- Python 3.11 or higher
- Poetry (recommended) or pip

### Install with Poetry (Recommended)
```bash
# Clone the repository
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI

# Install dependencies
poetry install
```

### Install with pip
```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
pip install -r requirements.txt
```

## 🎯 First Steps

### 1. Fetch Data
```bash
# Fetch data for all configured symbols
poetry run quanttradeai fetch-data
```

This will download OHLCV data for AAPL, META, TSLA, JPM, and AMZN and cache it locally.

### 2. Run Training Pipeline
```bash
# Run the complete training pipeline
poetry run quanttradeai train
```

This will:
- Process the data and generate features
- Train ensemble models for each symbol
- Optimize hyperparameters automatically
- Save trained models

Tip: To control the test window, set `data.test_start` (and optional `data.test_end`) in `config/model_config.yaml`. Hyperparameter tuning uses time‑series cross‑validation (`cv_folds`).

### 3. Evaluate Results
```bash
# Evaluate a trained model
poetry run quanttradeai evaluate -m models/trained/AAPL
```

### 4. Backtest a Saved Model
```bash
# Backtest a saved model on the configured test window (with execution costs)
poetry run quanttradeai backtest-model -m models/experiments/<timestamp>/<SYMBOL> \
  -c config/model_config.yaml -b config/backtest_config.yaml
```

This runs an end-to-end evaluation using the model’s saved `feature_columns` and the execution configuration. Artifacts are saved under `reports/backtests/<run_timestamp>/<SYMBOL>/`:
- metrics.json: summary including Sharpe, drawdown, and CAGR (gross and net)
- equity_curve.csv: equity curve time series
- ledger.csv: per-trade fills and costs (when trades occur)

You'll also find a consolidated `reports/backtests/<run_timestamp>/portfolio/` folder containing portfolio-level metrics and equity curve that aggregate every successful symbol in the run.

## 📊 Understanding the Output

After running the training pipeline, you'll find:

- **Trained Models**: `models/trained/` - Saved models for each symbol
- **Experiment Results**: `models/experiments/` - Training logs and metrics
- **Cached Data**: `data/raw/` - Downloaded OHLCV data
- **Processed Data**: `data/processed/` - Feature-engineered data

## 🔧 Configuration

The framework uses two main configuration files:

- **`config/model_config.yaml`** - Model parameters and data settings
- **`config/features_config.yaml`** - Feature engineering settings

See the [Configuration Guide](configuration.md) for detailed settings.

## 📚 Next Steps

- **[Quick Reference](quick-reference.md)** - Common commands and patterns
- **[API Reference](api/)** - Detailed API documentation

- **[Configuration](configuration.md)** - Configuration options

## 🆘 Troubleshooting

### Common Issues

**Data fetching fails:**
```bash
# Force refresh data
poetry run quanttradeai fetch-data --refresh
```

**Training takes too long:**
- Reduce `n_trials` in hyperparameter optimization
- Use fewer symbols in configuration

**Memory issues:**
- Reduce the date range in configuration
- Process symbols one at a time

### Getting Help

- Check the [Quick Reference](quick-reference.md) for common patterns
- Review the [API Reference](api/) for detailed function documentation
- See the [Quick Reference](quick-reference.md) for complete workflow examples
- Open an [issue](https://github.com/AKKI0511/QuantTradeAI/issues) for bugs

## 🎉 Congratulations!

You've successfully set up QuantTradeAI! You can now:

- Train models for different assets
- Experiment with different features
- Backtest trading strategies
- Deploy models for production use

Happy trading! 🚀
