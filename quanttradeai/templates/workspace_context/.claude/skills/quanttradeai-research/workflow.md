# QuantTradeAI Workflow

## 1. Inspect The Workspace

Start by reading:

* `config/project.yaml`
* `AGENTS.md`
* relevant skill files if needed

Identify:

* symbols
* date windows
* timeframe
* features
* agents
* sweeps
* risk settings
* deployment mode

## 2. Edit YAML First

Prefer changing `config/project.yaml`.

Good YAML edits include:

* changing symbols/date windows
* adding or adjusting feature definitions
* adding rule/model/LLM/hybrid agents
* adding sweeps
* changing risk limits
* changing paper/backtest settings

Avoid writing custom scripts unless QuantTradeAI cannot express the requested workflow.

## 3. Validate After Every YAML Change

```bash
quanttradeai validate -c config/project.yaml
```

If validation fails, fix the YAML before running experiments.

## 4. Run Strategy Backtests

To run every configured agent:

```bash
quanttradeai agent run --all -c config/project.yaml --mode backtest --max-concurrency 4
```

To run one agent:

```bash
quanttradeai agent run --agent <agent_name> -c config/project.yaml --mode backtest
```

## 5. Run Sweeps

Use sweeps for parameter search.

```bash
quanttradeai agent run --sweep <sweep_name> -c config/project.yaml --mode backtest --max-concurrency 4
```

Sweeps are backtest-only in the current product stage.

## 6. Rank Runs

Use the scoreboard before selecting a winner.

```bash
quanttradeai runs list --scoreboard --sort-by net_sharpe
```

Default ranking:

* backtest: `net_sharpe`
* paper/live: `total_pnl`

If the human specifies a goal like lower drawdown, higher PnL, or fewer trades, use that goal when judging results.

## 7. Compare Top Runs

Compare finalists before recommending a winner.

```bash
quanttradeai runs list --compare agent/backtest/<run_a> --compare agent/backtest/<run_b> --sort-by net_sharpe
```

Review metric differences and config deltas.

## 8. Promote A Winner

Promote only successful runs after artifact review.

```bash
quanttradeai promote --run agent/backtest/<winning_run_id> -c config/project.yaml
```

For research model promotion:

```bash
quanttradeai promote --run research/<winning_run_id> -c config/project.yaml
```

## 9. Run Paper Mode

After backtest promotion, use paper mode.

```bash
quanttradeai agent run --agent <agent_name> -c config/project.yaml --mode paper
```

Prefer replay-backed paper mode unless the human explicitly asks for broker-backed execution.

## 10. Deployment Preparation

To generate a deployment bundle:

```bash
quanttradeai deploy --agent <agent_name> -c config/project.yaml --target local
quanttradeai deploy --agent <agent_name> -c config/project.yaml --target docker-compose
quanttradeai deploy --agent <agent_name> -c config/project.yaml --target render
```

Generating a deployment bundle is not the same as starting live trading.

If the human asks to "deploy", clarify whether they mean:

* generate a bundle
* run paper
* promote to live
* actually start live trading

## 11. Final Report

When done, report:

* what was changed
* commands run
* artifact locations
* winning run
* key metrics
* closest alternatives
* risks/caveats
* next recommended step
