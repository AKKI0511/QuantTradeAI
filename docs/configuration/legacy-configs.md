# Legacy Config Compatibility

Legacy YAMLs are still supported, but they are no longer the best starting point for a new project.

If you are already running QuantTradeAI with the older `config/` layout, this page shows the shortest safe path forward.

## Files Still Supported

| File | Still used by |
| --- | --- |
| `config/model_config.yaml` | `train`, `evaluate`, `backtest-model`, `live-trade` |
| `config/features_config.yaml` | `train`, `evaluate`, `live-trade`, saved-model backtests |
| `config/backtest_config.yaml` | `backtest`, `backtest-model` |
| `config/impact_config.yaml` | `backtest-model`, validator |
| `config/risk_config.yaml` | `backtest-model`, `live-trade` |
| `config/streaming.yaml` | `live-trade` |
| `config/position_manager.yaml` | `live-trade` |

## Best Migration Path

If you want to keep your existing files but move toward the canonical flow:

```bash
poetry run quanttradeai validate --legacy-config-dir config
```

This does three useful things:

- loads your existing legacy YAMLs
- synthesizes a canonical project config
- writes the migrated artifact into the validation output

From there, you can review the migrated file and turn it into a real `config/project.yaml`.

## Legacy Commands That Still Matter

```bash
poetry run quanttradeai train -c config/model_config.yaml
poetry run quanttradeai evaluate -m models/experiments/<timestamp>/<SYMBOL> -c config/model_config.yaml
poetry run quanttradeai backtest-model -m models/experiments/<timestamp>/<SYMBOL> -c config/model_config.yaml -b config/backtest_config.yaml
poetry run quanttradeai live-trade -m models/experiments/<timestamp>/<SYMBOL> -c config/model_config.yaml -s config/streaming.yaml
```

## What to Prefer for New Work

- For research and agent backtests: use [`config/project.yaml`](project-yaml.md)
- For live trading and execution runtime: keep using the runtime YAMLs in [`Runtime and Live Trading Configs`](live-runtime-files.md)

## Bottom Line

Treat the legacy YAML set as a compatibility layer:

- useful when you already have a working setup
- still required for live trading today
- not the best place to start a new research or agent workflow
