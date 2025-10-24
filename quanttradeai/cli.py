"""Typer-based production CLI for QuantTradeAI.

Commands mirror the legacy argparse CLI and add `backtest-model`.
The console script still points to `quanttradeai.main:main` which delegates
here, so developer workflow remains unchanged.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import typer

from .main import (
    run_pipeline,
    fetch_data_only,
    evaluate_model,
    run_live_pipeline,
    run_model_backtest,
)
from .backtest.backtester import simulate_trades, compute_metrics
import yaml
import pandas as pd


app = typer.Typer(add_completion=False, help="QuantTradeAI command line interface")


@app.command("fetch-data")
def cmd_fetch_data(
    config: str = typer.Option(
        "config/model_config.yaml", "-c", "--config", help="Path to config file"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Force refresh of cache"),
):
    """Fetch OHLCV data and cache it."""

    fetch_data_only(config, refresh)


@app.command("train")
def cmd_train(
    config: str = typer.Option(
        "config/model_config.yaml", "-c", "--config", help="Path to config file"
    )
):
    """Run full training pipeline."""

    run_pipeline(config)


@app.command("evaluate")
def cmd_evaluate(
    model_path: str = typer.Option(..., "-m", "--model-path", help="Saved model dir"),
    config: str = typer.Option(
        "config/model_config.yaml", "-c", "--config", help="Path to config file"
    ),
):
    """Evaluate a saved model on current dataset."""

    evaluate_model(config, model_path)


@app.command("backtest")
def cmd_backtest(
    config: str = typer.Option(
        "config/backtest_config.yaml", "-c", "--config", help="Backtest config file"
    ),
    cost_bps: Optional[float] = typer.Option(None, help="Transaction cost in bps"),
    cost_fixed: Optional[float] = typer.Option(None, help="Fixed transaction cost"),
    slippage_bps: Optional[float] = typer.Option(None, help="Slippage in bps"),
    slippage_fixed: Optional[float] = typer.Option(None, help="Fixed slippage amount"),
    liquidity_max_participation: Optional[float] = typer.Option(
        None, help="Liquidity max participation"
    ),
):
    """Run backtest using a CSV and backtest execution config."""

    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    exec_cfg = cfg.get("execution", {})
    if cost_bps is not None:
        exec_cfg.setdefault("transaction_costs", {})
        exec_cfg["transaction_costs"].update(
            {"enabled": True, "mode": "bps", "value": cost_bps}
        )
    if cost_fixed is not None:
        exec_cfg.setdefault("transaction_costs", {})
        exec_cfg["transaction_costs"].update(
            {"enabled": True, "mode": "fixed", "value": cost_fixed}
        )
    if slippage_bps is not None:
        exec_cfg.setdefault("slippage", {})
        exec_cfg["slippage"].update(
            {"enabled": True, "mode": "bps", "value": slippage_bps}
        )
    if slippage_fixed is not None:
        exec_cfg.setdefault("slippage", {})
        exec_cfg["slippage"].update(
            {"enabled": True, "mode": "fixed", "value": slippage_fixed}
        )
    if liquidity_max_participation is not None:
        exec_cfg.setdefault("liquidity", {})
        exec_cfg["liquidity"].update(
            {"enabled": True, "max_participation": liquidity_max_participation}
        )
    df = pd.read_csv(cfg["data_path"])
    result = simulate_trades(df, execution=exec_cfg)
    metrics = compute_metrics(result)
    typer.echo(json.dumps(metrics, indent=2))


@app.command("backtest-model")
def cmd_backtest_model(
    model_path: str = typer.Option(..., "-m", "--model-path", help="Saved model dir"),
    config: str = typer.Option(
        "config/model_config.yaml", "-c", "--config", help="Model config file"
    ),
    backtest_config: Optional[str] = typer.Option(
        "config/backtest_config.yaml", "-b", "--backtest-config", help="Backtest config"
    ),
    risk_config: Optional[str] = typer.Option(
        "config/risk_config.yaml",
        "--risk-config",
        help="Risk management config applied via drawdown guard",
    ),
    cost_bps: Optional[float] = typer.Option(None, help="Transaction cost in bps"),
    cost_fixed: Optional[float] = typer.Option(None, help="Fixed transaction cost"),
    slippage_bps: Optional[float] = typer.Option(None, help="Slippage in bps"),
    slippage_fixed: Optional[float] = typer.Option(None, help="Fixed slippage amount"),
    liquidity_max_participation: Optional[float] = typer.Option(
        None, help="Liquidity max participation"
    ),
):
    """Backtest a saved model on the configured test window with execution costs."""

    summary = run_model_backtest(
        model_config=config,
        model_path=model_path,
        backtest_config=backtest_config,
        risk_config=risk_config,
        cost_bps=cost_bps,
        cost_fixed=cost_fixed,
        slippage_bps=slippage_bps,
        slippage_fixed=slippage_fixed,
        liquidity_max_participation=liquidity_max_participation,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("live-trade")
def cmd_live_trade(
    url: str = typer.Option(..., "--url", help="WebSocket URL for streaming data"),
    config: str = typer.Option(
        "config/model_config.yaml", "-c", "--config", help="Path to config file"
    ),
):
    """Run real-time trading pipeline using streaming input."""

    asyncio.run(run_live_pipeline(config, url))


if __name__ == "__main__":
    app()
