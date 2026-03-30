"""Typer-based production CLI for QuantTradeAI.

Commands mirror the legacy argparse CLI and add `backtest-model`.
The console script points directly to `quanttradeai.cli:app`, while
`quanttradeai.main:main` remains as a legacy bridge for older entry paths.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
from .utils.config_validator import (
    DEFAULT_CONFIG_PATHS,
    validate_all,
    validate_project_config,
)
from .utils.project_paths import infer_project_root
from .utils.project_config import (
    compile_research_runtime_configs,
    load_project_config,
)
from .utils.run_records import (
    RunFilters,
    apply_required_run_fields,
    create_run_dir,
    discover_runs,
    filter_runs,
)


app = typer.Typer(add_completion=False, help="QuantTradeAI command line interface")
research_app = typer.Typer(help="Research workflows")
agent_app = typer.Typer(help="Agent workflows")
runs_app = typer.Typer(help="Run records")
app.add_typer(research_app, name="research")
app.add_typer(agent_app, name="agent")
app.add_typer(runs_app, name="runs")


def fetch_data_only(*args, **kwargs):
    from .main import fetch_data_only as _fetch_data_only

    return _fetch_data_only(*args, **kwargs)


def run_pipeline(*args, **kwargs):
    from .main import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


def evaluate_model(*args, **kwargs):
    from .main import evaluate_model as _evaluate_model

    return _evaluate_model(*args, **kwargs)


async def run_live_pipeline(*args, **kwargs):
    from .main import run_live_pipeline as _run_live_pipeline

    return await _run_live_pipeline(*args, **kwargs)


def run_model_backtest(*args, **kwargs):
    from .main import run_model_backtest as _run_model_backtest

    return _run_model_backtest(*args, **kwargs)


def simulate_trades(*args, **kwargs):
    from .backtest.backtester import simulate_trades as _simulate_trades

    return _simulate_trades(*args, **kwargs)


def compute_metrics(*args, **kwargs):
    from .backtest.backtester import compute_metrics as _compute_metrics

    return _compute_metrics(*args, **kwargs)


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
            "model": {
                "kind": "classifier",
                "family": "voting",
                "tuning": {"enabled": True, "trials": 50},
            },
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
            "streaming": {
                "enabled": True,
                "provider": "alpaca",
                "websocket_url": "wss://stream.data.alpaca.markets/v2/iex",
                "auth_method": "api_key",
                "symbols": ["AAPL"],
                "channels": ["trades", "quotes"],
                "buffer_size": 1000,
                "reconnect_attempts": 5,
                "monitoring": {"enabled": True, "check_interval": 5},
                "metrics": {"enabled": False, "host": "0.0.0.0", "port": 9000},
                "api": {"enabled": False, "host": "0.0.0.0", "port": 8000},
            },
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
            "streaming": {
                "enabled": True,
                "provider": "alpaca",
                "websocket_url": "wss://stream.data.alpaca.markets/v2/iex",
                "auth_method": "api_key",
                "symbols": ["AAPL", "TSLA"],
                "channels": ["trades", "quotes"],
                "buffer_size": 1000,
                "reconnect_attempts": 5,
                "monitoring": {"enabled": True, "check_interval": 5},
                "metrics": {"enabled": False, "host": "0.0.0.0", "port": 9000},
                "api": {"enabled": False, "host": "0.0.0.0", "port": 8000},
            },
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
            "model": {
                "kind": "classifier",
                "family": "voting",
                "tuning": {"enabled": True, "trials": 50},
            },
            "evaluation": {"split": "time_aware", "use_configured_test_window": True},
            "backtest": {"costs": {"enabled": True, "bps": 5}},
        },
        "agents": [
            {
                "name": "hybrid_swing_agent",
                "kind": "hybrid",
                "mode": "paper",
                "model_signal_sources": [],
                "llm": {
                    "provider": "openai",
                    "model": "gpt-5.3",
                    "prompt_file": "prompts/hybrid_swing.md",
                },
                "context": {
                    "features": ["rsi_14"],
                    "model_signals": [],
                    "positions": True,
                },
                "tools": ["get_quote", "place_order"],
                "risk": {"max_position_pct": 0.05, "max_daily_loss_pct": 0.02},
            }
        ],
        "deployment": {"target": "docker-compose", "mode": "paper"},
    },
    "model-agent": {
        "project": {"name": "model_agent_lab", "profile": "paper"},
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
            "streaming": {
                "enabled": True,
                "provider": "alpaca",
                "websocket_url": "wss://stream.data.alpaca.markets/v2/iex",
                "auth_method": "api_key",
                "symbols": ["AAPL"],
                "channels": ["trades", "quotes"],
                "buffer_size": 1000,
                "reconnect_attempts": 5,
                "monitoring": {"enabled": True, "check_interval": 5},
                "metrics": {"enabled": False, "host": "0.0.0.0", "port": 9000},
                "api": {"enabled": False, "host": "0.0.0.0", "port": 8000},
            },
        },
        "features": {
            "definitions": [
                {"name": "rsi_14", "type": "technical", "params": {"period": 14}}
            ]
        },
        "research": {
            "enabled": False,
            "labels": {
                "type": "forward_return",
                "horizon": 5,
                "buy_threshold": 0.01,
                "sell_threshold": -0.01,
            },
            "model": {
                "kind": "classifier",
                "family": "voting",
                "tuning": {"enabled": False, "trials": 1},
            },
            "evaluation": {"split": "time_aware", "use_configured_test_window": True},
            "backtest": {"costs": {"enabled": True, "bps": 5}},
        },
        "agents": [
            {
                "name": "paper_momentum",
                "kind": "model",
                "mode": "paper",
                "model": {"path": "models/trained/aapl_daily_classifier"},
                "context": {
                    "features": ["rsi_14"],
                    "positions": True,
                    "risk_state": True,
                },
                "tools": ["get_quote", "place_order"],
                "risk": {"max_position_pct": 0.05},
            }
        ],
        "deployment": {"target": "docker-compose", "mode": "paper"},
    },
}

PROJECT_TEMPLATE_PROMPTS = {
    "llm-agent": {
        "prompts/breakout.md": """# Breakout Agent

Review the provided market data, selected features, current position state, and risk limits.
Bias toward holding unless the evidence for a directional trade is clear.
Use the available tools only as reference context; do not invent tool output.
""",
    },
    "hybrid": {
        "prompts/hybrid_swing.md": """# Hybrid Swing Agent

Review the provided market data, engineered features, current position state, risk limits, and model signals.
Treat model signals as one input, not an automatic order.
Use the available tools only as reference context; do not invent tool output.
""",
    },
    "model-agent": {
        "models/trained/aapl_daily_classifier/README.md": """# Placeholder Model Artifact

Replace this directory with a trained model artifact before running the model agent.
"""
    },
}


def _write_template_assets(template_name: str, output_path: Path, force: bool) -> None:
    assets = PROJECT_TEMPLATE_PROMPTS.get(template_name, {})
    if not assets:
        return

    project_root = infer_project_root(output_path)
    for relative_path, content in assets.items():
        asset_path = project_root / relative_path
        if asset_path.exists() and not force:
            continue
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_text(content.strip() + "\n", encoding="utf-8")


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

    import pandas as pd
    import yaml

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


def _project_to_runtime_configs(
    project_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_cfg, features_cfg, _ = compile_research_runtime_configs(
        project_config,
        require_research=False,
    )
    return model_cfg, features_cfg


def _copy_artifact(source: str | Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = Path(source).read_text(encoding="utf-8")
    destination.write_text(payload, encoding="utf-8")
    return str(destination)


def _normalize_choice(value: str, *, allowed: set[str], field_name: str) -> str:
    normalized = value.lower().strip()
    if normalized not in allowed:
        valid = ", ".join(sorted(allowed))
        raise typer.BadParameter(f"{field_name} must be one of: {valid}")
    return normalized


def _format_symbols(symbols: list[str]) -> str:
    if not symbols:
        return "-"
    if len(symbols) <= 3:
        return ",".join(symbols)
    return ",".join(symbols[:3]) + f"+{len(symbols) - 3}"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 3)] + "..."


def _render_runs_table(records: list[dict[str, Any]]) -> str:
    columns = [
        ("RUN_ID", 40),
        ("TYPE", 8),
        ("MODE", 10),
        ("STATUS", 8),
        ("NAME", 24),
        ("STARTED_AT", 25),
        ("SYMBOLS", 20),
    ]
    header = "  ".join(f"{label:<{width}}" for label, width in columns)
    lines = [header]
    for record in records:
        timestamps = dict(record.get("timestamps") or {})
        values = {
            "RUN_ID": str(record.get("run_id") or "-"),
            "TYPE": str(record.get("run_type") or "-"),
            "MODE": str(record.get("mode") or "-"),
            "STATUS": str(record.get("status") or "-"),
            "NAME": str(record.get("name") or "-"),
            "STARTED_AT": str(timestamps.get("started_at") or "-"),
            "SYMBOLS": _format_symbols(list(record.get("symbols") or [])),
        }
        lines.append(
            "  ".join(
                f"{_truncate(values[label], width):<{width}}"
                for label, width in columns
            )
        )
    return "\n".join(lines)


@research_app.command("run")
def cmd_research_run(
    config: str = typer.Option(
        "config/project.yaml", "-c", "--config", help="Path to project config YAML"
    ),
    legacy_config_dir: Optional[str] = typer.Option(
        None,
        "--legacy-config-dir",
        help="Import legacy YAML files from this directory instead of reading project.yaml",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip data-quality validation before training",
    ),
):
    """Run the Stage 1 canonical research workflow from project.yaml."""

    import yaml

    project_path = Path(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    try:
        loaded_project = load_project_config(
            config_path=project_path,
            legacy_config_dir=legacy_config_dir,
        )
        run_name = (
            (loaded_project.raw.get("project") or {}).get("name")
        ) or project_path.stem
        initial_warnings = list(loaded_project.warnings)
    except Exception:
        run_name = project_path.stem
        initial_warnings = []

    run_dir, run_id = create_run_dir(
        run_type="research",
        mode="research",
        name=run_name,
        timestamp=timestamp,
    )

    summary = {
        "project_name": None,
        "project_profile": None,
        "symbols": [],
        "feature_count": 0,
        "model_family": None,
        "timestamps": {"started_at": datetime.now(timezone.utc).isoformat()},
        "status": "failed",
        "run_dir": str(run_dir),
        "warnings": list(initial_warnings),
        "artifacts": {},
    }
    apply_required_run_fields(
        summary,
        run_dir=run_dir,
        run_type="research",
        mode="research",
        name=run_name,
    )
    summary["run_id"] = run_id

    try:
        validation = validate_project_config(
            config_path=project_path,
            output_dir=run_dir / "validation",
            legacy_config_dir=legacy_config_dir,
            timestamp_subdir=False,
        )
        summary["warnings"] = list(
            dict.fromkeys(summary["warnings"] + validation.get("warnings", []))
        )
        resolved_path = Path(validation["artifacts"]["resolved_config"])
        root_resolved_path = run_dir / "resolved_project_config.yaml"
        summary["artifacts"]["resolved_project_config"] = _copy_artifact(
            resolved_path,
            root_resolved_path,
        )
        summary["artifacts"]["validation_summary"] = validation["artifacts"]["summary"]
        if validation["artifacts"].get("migrated_project_config"):
            summary["artifacts"]["migrated_project_config"] = _copy_artifact(
                validation["artifacts"]["migrated_project_config"],
                run_dir / "migrated_project_config.yaml",
            )

        with root_resolved_path.open("r", encoding="utf-8") as handle:
            resolved_project = yaml.safe_load(handle) or {}

        missing = [
            name
            for name in ("data", "features", "research")
            if name not in resolved_project
        ]
        if missing:
            raise ValueError(
                "Project config missing required section(s) for research run: "
                + ", ".join(missing)
            )

        model_cfg, features_cfg, backtest_cfg = compile_research_runtime_configs(
            resolved_project
        )
        runtime_model_path = run_dir / "runtime_model_config.yaml"
        runtime_features_path = run_dir / "runtime_features_config.yaml"
        runtime_backtest_path = run_dir / "runtime_backtest_config.yaml"
        runtime_model_path.write_text(
            yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8"
        )
        runtime_features_path.write_text(
            yaml.safe_dump(features_cfg, sort_keys=False), encoding="utf-8"
        )
        runtime_backtest_path.write_text(
            yaml.safe_dump(backtest_cfg, sort_keys=False), encoding="utf-8"
        )
        summary["artifacts"]["runtime_model_config"] = str(runtime_model_path)
        summary["artifacts"]["runtime_features_config"] = str(runtime_features_path)
        summary["artifacts"]["runtime_backtest_config"] = str(runtime_backtest_path)

        training_cfg = model_cfg.get("training", {})
        pipeline_output = run_pipeline(
            str(runtime_model_path),
            skip_validation=skip_validation,
            include_metadata=True,
            features_config_path=str(runtime_features_path),
            tuning_enabled=bool(training_cfg.get("tune_hyperparameters", True)),
            optuna_trials=int(training_cfg.get("optuna_trials", 50)),
        )

        results = (
            pipeline_output.get("results", {})
            if isinstance(pipeline_output, dict)
            else {}
        )
        experiment_dir = (
            pipeline_output.get("experiment_dir")
            if isinstance(pipeline_output, dict)
            else None
        )

        summary.update(
            {
                "project_name": resolved_project.get("project", {}).get("name"),
                "project_profile": resolved_project.get("project", {}).get("profile"),
                "symbols": (resolved_project.get("data") or {}).get("symbols", []),
                "feature_count": len(
                    (resolved_project.get("features") or {}).get("definitions", [])
                ),
                "model_family": (
                    (resolved_project.get("research") or {}).get("model") or {}
                ).get("family"),
                "status": "success",
            }
        )
        if experiment_dir:
            summary["artifacts"]["experiment_dir"] = experiment_dir
            summary["artifacts"]["results"] = str(Path(experiment_dir) / "results.json")
            summary["artifacts"]["coverage"] = str(
                Path(experiment_dir) / "test_window_coverage.json"
            )
            summary["artifacts"]["preprocessing"] = str(
                Path(experiment_dir) / "preprocessing.json"
            )

        backtest_payload: dict[str, Any] = {}
        if experiment_dir:
            backtest_root = run_dir / "backtests"
            for symbol in sorted(results) or sorted(summary["symbols"]):
                model_path = Path(experiment_dir) / symbol
                if not model_path.is_dir():
                    summary["warnings"].append(
                        f"Skipping automatic backtest for {symbol}: model artifact directory not found at {model_path}."
                    )
                    continue

                symbol_backtest = run_model_backtest(
                    model_config=str(runtime_model_path),
                    model_path=str(model_path),
                    backtest_config=str(runtime_backtest_path),
                    risk_config=None,
                    skip_validation=skip_validation,
                    symbols=[symbol],
                    output_dir=backtest_root / symbol,
                )
                backtest_payload[symbol] = symbol_backtest

        if backtest_payload:
            backtest_summary_path = run_dir / "backtest_summary.json"
            backtest_summary_path.write_text(
                json.dumps(backtest_payload, indent=2),
                encoding="utf-8",
            )
            summary["artifacts"]["backtest_summary"] = str(backtest_summary_path)

        summary["warnings"] = list(dict.fromkeys(summary["warnings"]))
        summary["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        apply_required_run_fields(
            summary,
            run_dir=run_dir,
            run_type="research",
            mode="research",
            name=summary.get("project_name") or run_name,
        )

        backtest_metrics_by_symbol = {}
        for symbol, payload in backtest_payload.items():
            if not isinstance(payload, dict):
                continue

            symbol_payload = payload.get(symbol, {})
            if not isinstance(symbol_payload, dict):
                continue

            metrics = symbol_payload.get("metrics")
            if isinstance(metrics, dict) and metrics:
                backtest_metrics_by_symbol[symbol] = metrics

        metrics_payload = {
            "status": "available" if results or backtest_payload else "placeholder",
            "research_metrics_by_symbol": {
                symbol: details.get("test_metrics", {})
                for symbol, details in results.items()
                if isinstance(details, dict)
            },
            "backtest_metrics_by_symbol": backtest_metrics_by_symbol,
        }
        if (
            not metrics_payload["research_metrics_by_symbol"]
            and not metrics_payload["backtest_metrics_by_symbol"]
        ):
            metrics_payload["status"] = "placeholder"
            metrics_payload["message"] = (
                "No research or backtest metrics were produced by the research workflow."
            )

    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        apply_required_run_fields(
            summary,
            run_dir=run_dir,
            run_type="research",
            mode="research",
            name=summary.get("project_name") or run_name,
        )
        metrics_payload = {
            "status": "placeholder",
            "message": "Research pipeline did not complete successfully.",
            "error": str(exc),
        }

        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
        typer.echo(f"Research run failed: {exc}", err=True)
        raise typer.Exit(code=1)

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    typer.echo(f"Research run completed: {run_dir}")
    typer.echo(json.dumps(summary, indent=2))


@runs_app.command("list")
def cmd_runs_list(
    run_type: str = typer.Option(
        "all", "--type", help="Run type filter: all, research, or agent"
    ),
    mode: str = typer.Option(
        "all",
        "--mode",
        help="Run mode filter: all, research, backtest, paper, or live",
    ),
    status: str = typer.Option(
        "all", "--status", help="Run status filter: all, success, or failed"
    ),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum runs to show"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit normalized run records as JSON"
    ),
):
    """List normalized research and agent runs from the local runs directory."""

    filters = RunFilters(
        run_type=_normalize_choice(
            run_type, allowed={"all", "research", "agent"}, field_name="type"
        ),
        mode=_normalize_choice(
            mode,
            allowed={"all", "research", "backtest", "paper", "live"},
            field_name="mode",
        ),
        status=_normalize_choice(
            status,
            allowed={"all", "success", "failed"},
            field_name="status",
        ),
        limit=limit,
    )
    records = filter_runs(discover_runs(), filters)

    if json_output:
        typer.echo(json.dumps(records, indent=2))
        return

    if not records:
        typer.echo("No runs found.")
        return

    typer.echo(_render_runs_table(records))


@agent_app.command("run")
def cmd_agent_run(
    agent: str = typer.Option(..., "--agent", help="Agent name from project config"),
    config: str = typer.Option(
        "config/project.yaml", "-c", "--config", help="Path to project config YAML"
    ),
    mode: str = typer.Option(
        "backtest",
        "--mode",
        help="Execution mode. backtest and paper are implemented; live remains future work.",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip data-quality validation before backtesting",
    ),
):
    """Run a first-class project agent in backtest or paper mode."""

    from .agents.backtest import run_agent_backtest
    from .agents.model_agent import run_model_agent_backtest, run_model_agent_paper
    from .agents.paper import run_agent_paper

    try:
        loaded_project = load_project_config(config_path=config)
        agent_config = next(
            (
                dict(item)
                for item in loaded_project.raw.get("agents") or []
                if item.get("name") == agent
            ),
            None,
        )
        if agent_config is None:
            raise ValueError(f"Agent '{agent}' not found in project config.")

        configured_mode = str(agent_config.get("mode") or "").strip().lower()
        if configured_mode and configured_mode != mode:
            typer.echo(
                f"Warning: agent '{agent}' is configured with mode={configured_mode} but CLI requested mode={mode}; continuing with CLI mode.",
                err=True,
            )

        agent_kind = agent_config.get("kind")
        if agent_kind == "model":
            if mode == "backtest":
                summary = run_model_agent_backtest(
                    project_config_path=config,
                    agent_name=agent,
                    skip_validation=skip_validation,
                )
            elif mode == "paper":
                if skip_validation:
                    typer.echo(
                        "Warning: --skip-validation is ignored for model agent paper runs.",
                        err=True,
                    )
                summary = run_model_agent_paper(
                    project_config_path=config,
                    agent_name=agent,
                )
            else:
                raise ValueError(
                    "Model agents currently support only --mode backtest or --mode paper."
                )
        elif agent_kind in {"llm", "hybrid"}:
            if mode == "backtest":
                summary = run_agent_backtest(
                    project_config_path=config,
                    agent_name=agent,
                    mode=mode,
                    skip_validation=skip_validation,
                )
            elif mode == "paper":
                if skip_validation:
                    typer.echo(
                        "Warning: --skip-validation is ignored for llm/hybrid agent paper runs.",
                        err=True,
                    )
                summary = run_agent_paper(
                    project_config_path=config,
                    agent_name=agent,
                )
            else:
                raise ValueError(
                    "LLM and hybrid agents currently support only --mode backtest or --mode paper."
                )
        elif agent_kind == "rule":
            raise ValueError("Rule agents are not implemented yet.")
        else:
            raise ValueError(f"Unsupported agent kind: {agent_kind}")
    except Exception as exc:
        typer.echo(f"Agent run failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Agent run completed: {summary['run_dir']}")
    typer.echo(json.dumps(summary, indent=2))


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

    import yaml

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
    _write_template_assets(normalized, output_path, force)

    typer.echo(f"Wrote {normalized} template to {output_path}")


@app.command("validate")
def cmd_validate(
    config: str = typer.Option(
        "config/project.yaml", "-c", "--config", help="Path to project config YAML"
    ),
    legacy_config_dir: Optional[str] = typer.Option(
        None,
        "--legacy-config-dir",
        help="Import legacy YAML files from this directory instead of reading project.yaml",
    ),
):
    """Validate canonical project config and emit resolved artifacts."""

    try:
        result = validate_project_config(
            config_path=config,
            legacy_config_dir=legacy_config_dir,
        )
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
