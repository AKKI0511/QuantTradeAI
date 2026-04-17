# Execution Costs Example

Run a model agent backtest from `config/project.yaml`, then compare it with a second run after increasing `research.backtest.costs.bps`.

```bash
poetry run quanttradeai init --template model-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode backtest
```

To increase execution costs, edit `config/project.yaml` and raise:

```yaml
research:
  backtest:
    costs:
      enabled: true
      bps: 10
```

Then rerun the same backtest and compare the two runs:

```bash
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode backtest
poetry run quanttradeai runs list --compare agent/backtest/<run_id_a> --compare agent/backtest/<run_id_b>
```

Higher costs should reduce net Sharpe, total PnL, or both.
