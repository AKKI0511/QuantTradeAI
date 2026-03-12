<div align="center">

# QuantTradeAI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](LICENSE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Poetry](https://img.shields.io/badge/poetry-managed-%231C1C1C)](https://python-poetry.org/)

<i>Machine learning framework for momentum‑driven quantitative trading</i>

</div>

## Why QuantTradeAI

- Research and production toolkit for ML-based trading strategies
- Designed for quants, data scientists, and ML engineers
- Time‑aware evaluation, configurable features, reproducible experiments

## Key Features

- 📊 Data: YFinance/AlphaVantage loaders, caching, validation
- 🧪 Features: technical indicators, custom signals, optional LLM sentiment
- 🤖 Models: ensemble VotingClassifier (LR, RF, XGBoost) with Optuna tuning
- 📈 Backtesting: execution costs, slippage, liquidity limits, adaptive impact, intrabar fills, borrow fees
- 🛡️ Risk management: drawdown and turnover guards
- 📟 Live trading: real-time position manager with intraday risk controls
- 🔌 Streaming: plugin-ready provider framework with automatic discovery and health monitoring
- 🛠️ CLI: end‑to‑end pipeline, evaluation, and model backtest in one place

## Quickstart

1) Install
```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
poetry install
```

2) Run the pipeline
```bash
# Show commands
poetry run quanttradeai --help

# Canonical Stage 1 happy path
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml

# Import an existing legacy config/ directory into the canonical workflow
poetry run quanttradeai validate --legacy-config-dir config
poetry run quanttradeai research run --legacy-config-dir config
```

3) Evaluate and backtest a saved model
```bash
# Evaluate a trained model on current data
# (use a path under models/experiments/<timestamp>/<SYMBOL> or your own models/trained/<SYMBOL>)
poetry run quanttradeai evaluate -m models/experiments/<timestamp>/<SYMBOL> -c config/model_config.yaml

# Backtest a saved model on the configured test window (with execution costs and optional risk halts)
poetry run quanttradeai backtest-model -m models/experiments/<timestamp>/<SYMBOL> -c config/model_config.yaml -b config/backtest_config.yaml --risk-config config/risk_config.yaml
```

Artifacts are written to:
- `runs/research/<timestamp>_<project>/` (`resolved_project_config.yaml`, runtime YAMLs, `summary.json`, `metrics.json`, automatic backtest summary)
- `models/experiments/<timestamp>/` (trained symbol models + `results.json`)
- `reports/backtests/<timestamp>/<SYMBOL>/` (`metrics.json`, `equity_curve.csv`, `ledger.csv`) for standalone backtest commands

## Configuration

- `config/project.yaml`: canonical Stage 1 config for research, agents, and deployment targets
- `config/model_config.yaml`, `config/features_config.yaml`, `config/backtest_config.yaml`: legacy compatibility files that can be imported with `--legacy-config-dir`
- `config/risk_config.yaml`: optional drawdown and turnover guards for standalone backtests and live trading
- `config/streaming.yaml`: optional provider, auth, and subscription settings for live trading
- `config/position_manager.yaml`: optional live position tracking and impact parameters

Pass `--risk-config path/to/risk.yaml` to `poetry run quanttradeai backtest-model` to enforce the configured drawdown guard during CLI backtests. If the file is omitted or missing, the backtest proceeds without halts.

Time‑aware evaluation rules:
- If `data.test_start` and `data.test_end` set: train = dates < `test_start`; test = `test_start` ≤ dates ≤ `test_end`
- If only `data.test_start` set: train = dates < `test_start`; test = dates ≥ `test_start`
- Otherwise: last `training.test_size` fraction is used chronologically (no shuffle)

See docs for details: [Configuration Guide](docs/configuration.md), [Quick Reference](docs/quick-reference.md).

## CLI Commands

```bash
poetry run quanttradeai init --template research -o config/project.yaml           # Generate canonical happy-path project config
poetry run quanttradeai validate -c config/project.yaml                            # Validate canonical project config and write resolved artifacts
poetry run quanttradeai research run -c config/project.yaml                        # Run the canonical end-to-end Stage 1 research workflow
poetry run quanttradeai validate --legacy-config-dir config                        # Import legacy YAMLs into canonical validation
poetry run quanttradeai research run --legacy-config-dir config                    # Import legacy YAMLs and run the canonical research workflow

# Legacy compatibility path (still supported)
poetry run quanttradeai fetch-data -c config/model_config.yaml
poetry run quanttradeai train -c config/model_config.yaml
poetry run quanttradeai evaluate -m <model_dir> -c config/model_config.yaml
poetry run quanttradeai backtest -c config/backtest_config.yaml
poetry run quanttradeai backtest-model -m <model_dir> -c config/model_config.yaml -b config/backtest_config.yaml --risk-config config/risk_config.yaml
poetry run quanttradeai validate-config
poetry run quanttradeai live-trade -m <model_dir> -c config/model_config.yaml -s config/streaming.yaml
```

## Python API

```python
from quanttradeai import DataLoader, DataProcessor, MomentumClassifier

loader = DataLoader("config/model_config.yaml")
processor = DataProcessor("config/features_config.yaml")
clf = MomentumClassifier("config/model_config.yaml")

data = loader.fetch_data()
df = processor.process_data(data["AAPL"])          # feature pipeline
df = processor.generate_labels(df)                   # forward returns → labels
X, y = clf.prepare_data(df)
clf.train(X, y)
```

## Environment & Secrets

- Copy `.env.example` to `.env` and fill values:

```bash
cp .env.example .env
```

Required vs optional keys:
- Required only if the related feature/provider is used.
- Examples:
  - LLM Sentiment (provider-dependent):
    - `OPENAI_API_KEY` (required if `provider: openai`)
    - `ANTHROPIC_API_KEY` (required if `provider: anthropic`)
    - `HUGGINGFACE_API_KEY` (required if `provider: huggingface`)
  - Streaming providers with `auth_method: api_key`:
    - Alpaca: `ALPACA_API_KEY`, `ALPACA_API_SECRET` (required if used)

## LLM Sentiment (Optional)

Configure in `config/features_config.yaml`:
```yaml
sentiment:
  enabled: true
  provider: openai
  model: gpt-3.5-turbo
  api_key_env_var: OPENAI_API_KEY
```
Export the key and run the pipeline. A `sentiment_score` column is added when a `text` column exists. See [docs/llm-sentiment.md](docs/llm-sentiment.md).

## Streaming (Optional)

- CLI: `poetry run quanttradeai live-trade -m models/experiments/<run>/AAPL --config config/model_config.yaml --streaming-config config/streaming.yaml`
- YAML‑driven gateway via `config/streaming.yaml`:

```yaml
streaming:
  symbols: ["AAPL"]
  providers:
    - name: "alpaca"
      websocket_url: "wss://stream.data.alpaca.markets/v2/iex"
      auth_method: "api_key"
      subscriptions: ["trades", "quotes"]
  buffer_size: 1000
  reconnect_attempts: 3
  health_check_interval: 30
```

```python
from quanttradeai.streaming import StreamingGateway

gw = StreamingGateway("config/streaming.yaml")
gw.subscribe_to_trades(["AAPL"], lambda m: print("TRADE", m))
# gw.start_streaming()  # blocking
```

- Provider adapters are discovered dynamically via `quanttradeai.streaming.providers.ProviderDiscovery`,
  validated with `ProviderConfigValidator`, and monitored through `ProviderHealthMonitor`. See
  [docs/api/streaming.md](docs/api/streaming.md) for detailed provider configuration and health tooling.
- The live trading pipeline (`quanttradeai.streaming.live_trading.LiveTradingEngine`) combines the
  streaming gateway, feature generation, model inference, risk controls, and optional health API. Use
  `--health-api true` to expose `/health` and `/metrics` while streaming.

### Streaming Health Monitoring

Enable advanced monitoring by adding a `streaming_health` section to your config and,
optionally, starting the embedded REST server:

```yaml
streaming_health:
  monitoring:
    enabled: true
    check_interval: 5
  metrics:
    enabled: true
    host: "0.0.0.0"
    port: 9000
  thresholds:
    max_latency_ms: 100
    min_throughput_msg_per_sec: 50
    max_queue_depth: 5000
  alerts:
    enabled: true
    channels: ["log", "metrics"]
    escalation_threshold: 3
  api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
```

Query live status while streaming:

```bash
curl http://localhost:8000/health    # readiness probe
curl http://localhost:8000/status    # detailed metrics + incidents
curl http://localhost:8000/metrics   # Prometheus scrape
```

Common patterns:

- Tune `escalation_threshold` to control alert promotion.
- Increase `max_queue_depth` in high-volume environments.
- Set `circuit_breaker_timeout` to avoid thrashing unstable providers.
- Run the standalone metrics exporter (default `0.0.0.0:9000`) when you want Prometheus
  scraping without enabling the FastAPI health server; if both are enabled on the same
  host/port, the health API continues to serve `/metrics` and the exporter stays
  disabled to avoid port collisions.

## Project Layout

```
quanttradeai/          # Core package
├─ data/               # Data sources, loader, processor
├─ features/           # Technical & custom features
├─ models/             # MomentumClassifier & utilities
├─ backtest/           # Vectorized backtester + metrics
├─ trading/            # Risk & portfolio management
├─ streaming/          # WebSocket gateway
├─ utils/              # Config schemas, metrics, viz
config/                # YAML configs (model, features, backtest, streaming)
docs/                  # Guides, API, examples
tests/                 # Pytest suite mirroring package
```

## Development

```bash
poetry install --with dev
make format     # Black
make lint       # flake8
make test       # pytest
```

Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)

## Links

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api/)
- [Examples](docs/examples/)
- [Configuration](docs/configuration.md)
- [Quick Reference](docs/quick-reference.md)

## Roadmap

The roadmap lives in [roadmap.md](roadmap.md) (source of truth) to avoid drift.

Current direction is explicitly two-track:
- **ML training loop**: reproducible data → features → time-aware training/eval → backtests → promotion.
- **Trading agent deployment**: developer-built agents (strategies + tools + prompts) running in parallel, tracked as experiments, deployed via existing platforms (Docker/K8s/managed runners).

Stage 1 research status today:
- Canonical `config/project.yaml`, `init`, `validate`, legacy import, resolved-config artifacts, and end-to-end `research run` are implemented.
- First-class agent execution, deployment, promotion, and run-list UX remain roadmap items.

## License

MIT © Contributors — see [LICENSE](LICENSE)
