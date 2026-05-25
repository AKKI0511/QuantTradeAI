---
name: quanttradeai-research
description: Use QuantTradeAI CLI and config/project.yaml to research, backtest, sweep, compare, promote, paper trade, or prepare deployment for trading strategies. Trigger when the user asks to test strategy ideas, find the best strategy, optimize parameters, compare trading agents, inspect results, or move a successful backtest toward paper trading.
---

# QuantTradeAI Research Skill

Use this skill to operate a QuantTradeAI workspace through YAML and CLI commands.

## Primary Interface

* Edit `config/project.yaml`.
* Run `quanttradeai` CLI commands.
* Read generated JSON artifacts.
* Avoid custom scripts unless the workflow cannot be expressed through QuantTradeAI config.

## Default Loop

1. Inspect `config/project.yaml`.
2. Translate the user's research objective into symbols, features, agents, sweeps, or risk settings.
3. Run validation.
4. Run backtests, batches, or sweeps.
5. Inspect artifacts before making claims.
6. Compare top candidates.
7. Recommend a winner or next experiment.
8. Promote only after a successful run and artifact review.

## Start Here

For detailed commands and flow, read `workflow.md`.

For artifact interpretation, read `artifacts.md`.

For live-trading boundaries, read `safety.md`.

## Strong Defaults

* Use backtest mode for research.
* Use replay-backed paper mode only after successful backtest/promotion.
* Use `net_sharpe` as the default backtest ranking metric unless the human asks for another objective.
* Prefer simple reproducible YAML changes over complex one-off code.

## Output To The Human

Report:

* YAML changes made
* commands run
* pass/fail status
* best run metrics
* closest alternatives
* risks or caveats
* recommended next action
