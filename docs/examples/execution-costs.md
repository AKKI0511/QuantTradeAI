# Execution Costs Example

Run a simple backtest with and without execution costs and slippage.

```bash
poetry run quanttradeai backtest --config config/backtest_config.yaml
poetry run quanttradeai backtest --cost-bps 5 --slippage-bps 10
```

The second command applies 5 bps transaction costs and 10 bps slippage, producing lower Sharpe ratio and PnL compared to the gross run.
