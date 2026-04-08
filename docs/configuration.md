# Configuration

QuantTradeAI has one **canonical project config** for research and agent backtests, plus a set of **runtime YAML files** that still power live trading and some compatibility workflows.

> Start with `config/project.yaml` if you are building a new research or agent workflow.
> Use the runtime YAMLs when you are running `live-trade`, saved-model backtests, or migrating older setups.

## Choose the Right File

| What you want to do | Primary file(s) | Used by |
| --- | --- | --- |
| Run the canonical research workflow | `config/project.yaml` | `quanttradeai validate`, `quanttradeai research run` |
| Run or promote a project-defined agent | `config/project.yaml` | `quanttradeai agent run`, `quanttradeai promote` |
| Generate a project-agent deployment bundle | `config/project.yaml` | `quanttradeai deploy` |
| Run live streaming inference | `config/model_config.yaml`, `config/features_config.yaml`, `config/streaming.yaml`, `config/risk_config.yaml`, `config/position_manager.yaml` | `quanttradeai live-trade` |
| Backtest a saved model with execution costs | `config/model_config.yaml`, `config/backtest_config.yaml`, optional `config/risk_config.yaml`, optional `config/impact_config.yaml` | `quanttradeai backtest-model` |
| Validate the runtime YAML bundle | Runtime YAML files under `config/` | `quanttradeai validate-config` |
| Import an existing legacy config bundle into the canonical validator | Legacy YAML files under `config/` | `quanttradeai validate --legacy-config-dir ...` |

## Recommended Reading

- [Project Config (`project.yaml`)](configuration/project-yaml.md)
- [Runtime and Live Trading Configs](configuration/live-runtime-files.md)
- [Legacy Config Compatibility](configuration/legacy-configs.md)

## Typical Workflows

### Research

```bash
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai runs list
```

### Agent Backtest

```bash
poetry run quanttradeai init --template llm-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
poetry run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
```

### Agent Deployment

```bash
poetry run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target docker-compose
```

### Live Trading

```bash
poetry run quanttradeai live-trade \
  -m models/experiments/<timestamp>/<SYMBOL> \
  -c config/model_config.yaml \
  -s config/streaming.yaml \
  --risk-config config/risk_config.yaml \
  --position-manager-config config/position_manager.yaml
```

## Important Boundaries

- `config/project.yaml` is the center of gravity for **research**, **agent runs**, **promotion**, and **deployment generation**
- `quanttradeai live-trade` does **not** read `config/project.yaml` today
- `config/streaming.yaml`, `config/risk_config.yaml`, and `config/position_manager.yaml` are still first-class runtime files
- `quanttradeai validate-config` is the fastest way to catch malformed runtime YAMLs before running live or backtest commands

## If You Are Migrating

Legacy YAMLs are still supported, but they are no longer the best place to start. If you already have a working `config/` folder, see [Legacy Config Compatibility](configuration/legacy-configs.md) for the shortest path into the current workflow.
