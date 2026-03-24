# QuantTradeAI

QuantTradeAI is a YAML-first, CLI-first framework for quant research and trading agents.

It supports two connected workflows:

- Research: data -> features -> labels -> training -> evaluation -> backtest -> run records
- Agents: YAML-defined `model`, `llm`, and `hybrid` agents that reuse the same project data and feature definitions

The roadmap source of truth is [roadmap.md](roadmap.md).

## What Works Today

- Canonical `config/project.yaml` workflow with `init`, `validate`, `research run`, and `runs list`
- Time-aware preprocessing and evaluation in the research path
- Standardized local run records with resolved config snapshots and metrics
- `model` agents from `project.yaml` in `backtest` and `paper` mode
- `llm` and `hybrid` agents from `project.yaml` in `backtest` mode
- Legacy saved-model backtests and legacy `live-trade` workflow from runtime YAML files

Current support matrix:

| Workflow | Status |
| --- | --- |
| `research run` from `project.yaml` | Supported |
| `agent run` for `model` agents in `backtest` | Supported |
| `agent run` for `model` agents in `paper` | Supported |
| `agent run` for `llm` and `hybrid` agents in `backtest` | Supported |
| `agent run` for `llm` and `hybrid` agents in `paper` | Not yet implemented |
| `rule` agents | Not yet implemented |
| `deploy` / promotion UX | Roadmap work |

## Install

```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
poetry install --with dev
```

You can also build and install with `pip install .`, but the documented developer workflow uses Poetry.

## Quickstart

### Research Project

```bash
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai runs list
```

### Model Agent From `project.yaml`

```bash
poetry run quanttradeai init --template model-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml

# Replace models/trained/aapl_daily_classifier/ with a real trained model artifact

poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode paper
```

### LLM Agent Backtest

```bash
poetry run quanttradeai init --template llm-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
```

### Hybrid Agent Backtest

```bash
poetry run quanttradeai init --template hybrid -o config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai agent run --agent hybrid_swing_agent -c config/project.yaml --mode backtest
```

## Artifacts

Canonical runs write standardized local artifacts:

- `runs/research/<timestamp>_<project>/`
- `runs/agent/backtest/<timestamp>_<agent>/`
- `runs/agent/paper/<timestamp>_<agent>/`

Typical artifacts include:

- `resolved_project_config.yaml`
- runtime YAML snapshots used by the run
- `summary.json`
- `metrics.json`
- `equity_curve.csv` and `ledger.csv` for backtests when available
- `decisions.jsonl` for agent backtests
- `executions.jsonl` for paper model-agent runs

## Configuration

Primary config surfaces:

- `config/project.yaml`: canonical happy-path config for research and project-defined agents
- `config/model_config.yaml`, `config/features_config.yaml`, `config/backtest_config.yaml`: legacy compatibility and saved-model workflows
- `config/streaming.yaml`, `config/risk_config.yaml`, `config/position_manager.yaml`: legacy live-trading runtime YAMLs

Important boundary:

- `agent run --mode paper` for `model` agents compiles runtime YAMLs from `project.yaml`
- `live-trade` still uses the runtime YAML files directly and does not read `project.yaml`

## Legacy Commands

These still work and remain useful for compatibility:

```bash
poetry run quanttradeai fetch-data -c config/model_config.yaml
poetry run quanttradeai train -c config/model_config.yaml
poetry run quanttradeai evaluate -m <model_dir> -c config/model_config.yaml
poetry run quanttradeai backtest-model -m <model_dir> -c config/model_config.yaml -b config/backtest_config.yaml
poetry run quanttradeai live-trade -m <model_dir> -c config/model_config.yaml -s config/streaming.yaml
poetry run quanttradeai validate-config
```

## Development

```bash
poetry install --with dev
make format
make lint
make test
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Configuration Overview](docs/configuration.md)
- [Project YAML](docs/configuration/project-yaml.md)
- [Runtime and Live Trading Configs](docs/configuration/live-runtime-files.md)
- [Quick Reference](docs/quick-reference.md)
- [API Docs](docs/api/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
