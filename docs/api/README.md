# Python API Reference

These pages are for advanced users who want to import QuantTradeAI from Python after installing the package.

Most users should start with the CLI and a single project YAML file. The CLI keeps the research, backtest, agent, artifact, and validation workflows consistent. Use the Python API when you need custom notebooks, bespoke research scripts, strategy extensions, or QuantTradeAI components embedded in another system.

```python
from quanttradeai import DataLoader, DataProcessor, simulate_trades, compute_metrics
```

## Pages

| Page | Covers |
| --- | --- |
| [Data](data.md) | Data sources, loading, validation, feature processing entry points |
| [Features](features.md) | Technical indicators, custom feature helpers, sentiment scoring |
| [Models](models.md) | `MomentumClassifier` training, inference, evaluation, persistence |
| [Backtesting](backtesting.md) | Trade simulation, metrics, execution costs, market impact models |
| [Agents](agents.md) | Strategy interfaces, rule agents, project agent runners |
| [Trading](trading.md) | Portfolio, position sizing, stop-loss/take-profit, risk guards |
| [Streaming](streaming.md) | Streaming gateways, replay helpers, provider adapters, health monitoring |
| [Utils](utils.md) | Advanced project config, validation, runs, scoreboard, comparison, sweep helpers |

## Related Docs

- [CLI docs](../cli/)
- [Config docs](../config/)
- [Artifacts](../artifacts.md)
- [Examples](../examples/)
