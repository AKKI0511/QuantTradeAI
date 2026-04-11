# Execution Costs Example

Run a simple backtest with and without execution costs and slippage.

```bash
poetry run quanttradeai backtest-model -m models/experiments/<timestamp>/<SYMBOL> \
  -c config/model_config.yaml -b config/backtest_config.yaml
poetry run quanttradeai backtest-model -m models/experiments/<timestamp>/<SYMBOL> \
  -c config/model_config.yaml -b config/backtest_config.yaml \
  --cost-bps 5 --slippage-bps 10
```

The second command applies 5 bps transaction costs and 10 bps slippage, producing lower Sharpe ratio and PnL compared to the gross run.
