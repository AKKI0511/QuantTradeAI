"""Typer-based production CLI for QuantTradeAI.

Commands mirror the legacy argparse CLI and add `backtest-model`.
The console script still points to `quanttradeai.main:main` which delegates
here, so developer workflow remains unchanged.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
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
from .utils.config_validator import (
    DEFAULT_CONFIG_PATHS,
    validate_all,
    validate_project_config,
)
import yaml
import pandas as pd


app = typer.Typer(add_completion=False, help="QuantTradeAI command line interface")


PROJECT_TEMPLATES = {
    "research": {
        "project": {"name": "research_lab", "profile": "research"},
        "profiles": {
            "research": {"mode": "research"},
            "paper": {"mode": "paper"},
            "live": {"mode": "live"},
        },
        "data": {
            "symbols": ["AAPL", "MSFT"],
            "start_date": "2018-01-01",
            "end_date": "2024-12-31",
            "timeframe": "1d",
            "test_start": "2024-09-01",
            "test_end": "2024-12-31",
        },
        "features": {
            "definitions": [
                {"name": "rsi_14", "type": "technical", "params": {"period": 14}}
            ]
        },
        "research": {
            "enabled": True,
            "labels": {
                "type": "forward_return",
                "horizon": 5,
                "buy_threshold": 0.01,
                "sell_threshold": -0.01,
            },
            "model": {"kind": "classifier", "family": "voting"},
            "evaluation": {"split": "time_aware", "use_configured_test_window": True},
            "backtest": {"costs": {"enabled": True, "bps": 5}},
        },
        "agents": [],
        "deployment": {"target": "docker-compose", "mode": "paper"},
    },
    "llm-agent": {
        "project": {"name": "agent_lab", "profile": "paper"},
        "profiles": {
            "research": {"mode": "research"},
            "paper": {"mode": "paper"},
            "live": {"mode": "live"},
        },
        "data": {
            "symbols": ["AAPL"],
            "start_date": "2022-01-01",
            "end_date": "2024-12-31",
            "timeframe": "1d",
            "test_start": "2024-09-01",
            "test_end": "2024-12-31",
        },
        "features": {
            "definitions": [
                {"name": "rsi_14", "type": "technical", "params": {"period": 14}}
            ]
        },
        "research": {
            "enabled": False,
            "labels": {},
            "model": {},
            "evaluation": {},
            "backtest": {},
        },
        "agents": [
            {
                "name": "breakout_gpt",
                "kind": "llm",
                "mode": "paper",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-5.3",
                    "prompt_file": "prompts/breakout.md",
                },
                "context": {
                    "features": ["rsi_14"],
                    "positions": True,
                    "risk_state": True,
                },
                "tools": ["get_quote", "get_position", "place_order"],
                "risk": {"max_position_pct": 0.05, "max_daily_loss_pct": 0.02},
            }
        ],
        "deployment": {"target": "docker-compose", "mode": "paper"},
    },
    "hybrid": {
        "project": {"name": "hybrid_lab", "profile": "paper"},
        "profiles": {
            "research": {"mode": "research"},
            "paper": {"mode": "paper"},
            "live": {"mode": "live"},
        },
        "data": {
            "symbols": ["AAPL", "TSLA"],
            "start_date": "2018-01-01",
            "end_date": "2024-12-31",
            "timeframe": "1d",
            "test_start": "2024-09-01",
            "test_end": "2024-12-31",
        },
        "features": {
            "definitions": [
                {"name": "rsi_14", "type": "technical", "params": {"period": 14}},
                {"name": "volume_spike_20", "type": "custom", "params": {"window": 20}},
            ]
        },
        "research": {
            "enabled": True,
            "labels": {
                "type": "forward_return",
                "horizon": 5,
                "buy_threshold": 0.01,
                "sell_threshold": -0.01,
            },
            "model": {"kind": "classifier", "family": "voting"},
            "evaluation": {"split": "time_aware", "use_configured_test_window": True},
            "backtest": {"costs": {"enabled": True, "bps": 5}},
        },
        "agents": [
            {
                "name": "hybrid_swing_agent",
                "kind": "hybrid",
                "mode": "paper",
                "model_signal_sources": ["aapl_daily_classifier"],
                "llm": {
                    "provider": "openai",
                    "model": "gpt-5.3",
                    "prompt_file": "prompts/hybrid_swing.md",
                },
                "context": {
                    "features": ["rsi_14"],
                    "model_signals": ["aapl_daily_classifier"],
                    "positions": True,
                },
                "tools": ["get_quote", "place_order"],
                "risk": {"max_position_pct": 0.05, "max_daily_loss_pct": 0.02},
            }
        ],
        "deployment": {"target": "docker-compose", "mode": "paper"},
    },
}


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
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip data-quality validation before training",
    ),
):
    """Run full training pipeline."""
    pipeline_result = run_pipeline(
        config, skip_validation=skip_validation, include_coverage=True
    )
    coverage_info = None
    if isinstance(pipeline_result, tuple) and len(pipeline_result) == 2:
        _, coverage_info = pipeline_result

    if coverage_info:
        fallback = coverage_info.get("fallback_symbols") or []
        path = coverage_info.get("path")
        summary = f"Test-window coverage report saved to {path}."
        if fallback:
            summary += " Fallback chronological split used for: " + ", ".join(fallback)
        typer.echo(summary, err=True)


@app.command("evaluate")
def cmd_evaluate(
    model_path: str = typer.Option(..., "-m", "--model-path", help="Saved model dir"),
    config: str = typer.Option(
        "config/model_config.yaml", "-c", "--config", help="Path to config file"
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip data-quality validation before evaluation",
    ),
):
    """Evaluate a saved model on current dataset."""

    evaluate_model(config, model_path, skip_validation=skip_validation)


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
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip data-quality validation before backtesting",
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
        skip_validation=skip_validation,
    )
    coverage_info = (
        summary.get("coverage_report") if isinstance(summary, dict) else None
    )
    if coverage_info:
        fallback = coverage_info.get("fallback_symbols") or []
        path = coverage_info.get("path")
        message = f"Test-window coverage report saved to {path}."
        if fallback:
            message += " Fallback chronological split used for: " + ", ".join(fallback)
        typer.echo(message, err=True)
    typer.echo(json.dumps(summary, indent=2))


@app.command("live-trade")
def cmd_live_trade(
    model_path: str = typer.Option(..., "-m", "--model-path", help="Saved model dir"),
    model_config: str = typer.Option(
        "config/model_config.yaml",
        "-c",
        "--config",
        help="Path to model configuration file",
    ),
    streaming_config: str = typer.Option(
        "config/streaming.yaml",
        "-s",
        "--streaming-config",
        help="Path to streaming configuration file",
    ),
    risk_config: Optional[str] = typer.Option(
        "config/risk_config.yaml",
        "--risk-config",
        help="Risk management configuration (drawdown/turnover)",
    ),
    position_manager_config: Optional[str] = typer.Option(
        "config/position_manager.yaml",
        "--position-manager-config",
        help="Position manager configuration for live execution",
    ),
    health_api: Optional[bool] = typer.Option(
        None,
        "--health-api",
        help="Override health API enable flag in streaming config (true/false)",
    ),
    initial_capital: float = typer.Option(
        1_000_000.0, "--initial-capital", help="Starting capital for sizing"
    ),
    history_window: int = typer.Option(
        512, "--history-window", help="History window for feature generation"
    ),
    min_history_for_features: int = typer.Option(
        220,
        "--min-history",
        help="Minimum bars required before running inference",
    ),
    stop_loss_pct: float = typer.Option(
        0.01, "--stop-loss-pct", help="Stop loss percentage applied to new trades"
    ),
):
    """Run real-time trading pipeline using streaming input."""

    asyncio.run(
        run_live_pipeline(
            model_config=model_config,
            model_path=model_path,
            streaming_config=streaming_config,
            risk_config=risk_config,
            position_manager_config=position_manager_config,
            enable_health_api=health_api,
            initial_capital=initial_capital,
            history_window=history_window,
            min_history_for_features=min_history_for_features,
            stop_loss_pct=stop_loss_pct,
        )
    )


@app.command("validate-config")
def cmd_validate_config(
    model_config: str = typer.Option(
        DEFAULT_CONFIG_PATHS["model_config"],
        "--model-config",
        help="Path to model configuration YAML",
    ),
    features_config: str = typer.Option(
        DEFAULT_CONFIG_PATHS["features_config"],
        "--features-config",
        help="Path to features configuration YAML",
    ),
    backtest_config: str = typer.Option(
        DEFAULT_CONFIG_PATHS["backtest_config"],
        "--backtest-config",
        help="Path to backtest configuration YAML",
    ),
    impact_config: str = typer.Option(
        DEFAULT_CONFIG_PATHS["impact_config"],
        "--impact-config",
        help="Path to impact configuration YAML",
    ),
    risk_config: str = typer.Option(
        DEFAULT_CONFIG_PATHS["risk_config"],
        "--risk-config",
        help="Path to risk configuration YAML",
    ),
    streaming_config: str = typer.Option(
        DEFAULT_CONFIG_PATHS["streaming_config"],
        "--streaming-config",
        help="Path to streaming configuration YAML",
    ),
    position_manager_config: str = typer.Option(
        DEFAULT_CONFIG_PATHS["position_manager_config"],
        "--position-manager-config",
        help="Path to position manager configuration YAML",
    ),
    output_dir: str = typer.Option(
        "reports/config_validation",
        "--output-dir",
        help="Directory to write validation summaries",
    ),
):
    """Validate all configuration files and emit a consolidated report."""

    summary = validate_all(
        {
            "model_config": model_config,
            "features_config": features_config,
            "backtest_config": backtest_config,
            "impact_config": impact_config,
            "risk_config": risk_config,
            "streaming_config": streaming_config,
            "position_manager_config": position_manager_config,
        },
        output_dir=output_dir,
    )
    typer.echo(json.dumps(summary, indent=2))
    if not summary.get("all_passed", False):
        raise typer.Exit(code=1)


@app.command("init")
def cmd_init(
    template: str = typer.Option(
        ..., "--template", help="Project template to initialize"
    ),
    output: str = typer.Option(
        "config/project.yaml",
        "-o",
        "--output",
        help="Path for generated project config",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing file"),
):
    """Initialize a canonical project config for the happy path."""

    normalized = template.lower()
    if normalized not in PROJECT_TEMPLATES:
        valid = ", ".join(sorted(PROJECT_TEMPLATES))
        raise typer.BadParameter(f"template must be one of: {valid}")

    output_path = Path(output)
    if output_path.exists() and not force:
        typer.echo(f"Refusing to overwrite existing file: {output_path}", err=True)
        raise typer.Exit(code=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(PROJECT_TEMPLATES[normalized], handle, sort_keys=False)

    typer.echo(f"Wrote {normalized} template to {output_path}")


@app.command("validate")
def cmd_validate(
    config: str = typer.Option(
        "config/project.yaml", "-c", "--config", help="Path to project config YAML"
    ),
):
    """Validate canonical project config and emit resolved artifacts."""

    try:
        result = validate_project_config(config_path=config)
    except Exception as exc:
        typer.echo(f"Project config validation failed: {exc}", err=True)
        raise typer.Exit(code=1)

    summary = result["summary"]
    typer.echo("Resolved project config summary:")
    typer.echo(
        f"- project: {summary['project']['name']} ({summary['project']['profile']})"
    )
    typer.echo(
        f"- data: symbols={len(summary['data']['symbols'])}, timeframe={summary['data']['timeframe']}, "
        f"range={summary['data']['date_range']['start']}..{summary['data']['date_range']['end']}"
    )
    typer.echo(
        f"- features: definitions={summary['feature_definitions']}, "
        f"research_enabled={summary['research_enabled']}, agents={len(summary['agents'])}"
    )

    if result.get("warnings"):
        for warning in result["warnings"]:
            typer.echo(f"Warning: {warning}", err=True)

    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
