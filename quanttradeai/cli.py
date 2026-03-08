"""Typer-based production CLI for QuantTradeAI.

Commands mirror the legacy argparse CLI and add `backtest-model`.
The console script still points to `quanttradeai.main:main` which delegates
here, so developer workflow remains unchanged.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
research_app = typer.Typer(help="Research workflows")
app.add_typer(research_app, name="research")


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


def _project_to_runtime_configs(
    project_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_cfg = dict(project_config.get("data") or {})
    research_cfg = dict(project_config.get("research") or {})
    features_cfg = dict(project_config.get("features") or {})

    def _or_default(value: Any, default: Any) -> Any:
        return default if value is None else value

    if not research_cfg.get("enabled", True):
        raise ValueError(
            "research.enabled must be true for `quanttradeai research run`."
        )

    model_cfg = {
        "data": {
            "symbols": data_cfg.get("symbols", []),
            "start_date": data_cfg.get("start_date"),
            "end_date": data_cfg.get("end_date"),
            "timeframe": _or_default(data_cfg.get("timeframe"), "1d"),
            "test_start": data_cfg.get("test_start"),
            "test_end": data_cfg.get("test_end"),
            "cache_dir": _or_default(data_cfg.get("cache_dir"), "data/raw"),
            "cache_path": _or_default(data_cfg.get("cache_path"), "data/raw"),
            "cache_expiration_days": _or_default(
                data_cfg.get("cache_expiration_days"), 7
            ),
            "use_cache": _or_default(data_cfg.get("use_cache"), True),
            "refresh": _or_default(data_cfg.get("refresh"), False),
            "max_workers": _or_default(data_cfg.get("max_workers"), 1),
        },
        "news": {
            "enabled": False,
            "provider": "yfinance",
            "lookback_days": 30,
            "symbols": [],
        },
        "models": {
            "family": research_cfg.get("model", {}).get("family", "voting"),
            "kind": research_cfg.get("model", {}).get("kind", "classifier"),
        },
        "labels": {
            "horizon": research_cfg.get("labels", {}).get("horizon", 5),
            "buy_threshold": research_cfg.get("labels", {}).get("buy_threshold", 0.01),
            "sell_threshold": research_cfg.get("labels", {}).get(
                "sell_threshold", -0.01
            ),
        },
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
        },
        "trading": {
            "initial_capital": 100000,
            "position_size": 0.2,
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "max_positions": 5,
            "transaction_cost": 0.001,
            "max_risk_per_trade": 0.02,
            "max_portfolio_risk": 0.10,
        },
    }

    feature_definitions = features_cfg.get("definitions") or []
    feature_steps = ["generate_technical_indicators"]
    price_features: list[str] = []
    momentum_features: dict[str, Any] = {}
    custom_features: list[dict[str, Any]] = []

    def _coerce_periods(raw_params: dict[str, Any]) -> list[int]:
        raw_periods = (
            raw_params.get("periods")
            or raw_params.get("lookback")
            or raw_params.get("period")
            or raw_params.get("window")
        )
        if raw_periods is None:
            return []
        if isinstance(raw_periods, list):
            return [int(value) for value in raw_periods]
        return [int(raw_periods)]

    def _resolve_custom_feature_key(name: str, params: dict[str, Any]) -> str:
        explicit = params.get("kind")
        if explicit in {
            "price_momentum",
            "volume_momentum",
            "mean_reversion",
            "volatility_breakout",
        }:
            return str(explicit)

        normalized_name = name.lower().strip()
        if normalized_name in {
            "price_momentum",
            "volume_momentum",
            "mean_reversion",
            "volatility_breakout",
        }:
            return normalized_name
        if normalized_name.startswith("volume_"):
            return "volume_momentum"
        if normalized_name.startswith("price_"):
            return "price_momentum"
        if "reversion" in normalized_name:
            return "mean_reversion"
        if "breakout" in normalized_name:
            return "volatility_breakout"

        raise ValueError(
            "features.definitions custom feature '"
            f"{name}' must map to one of: price_momentum, volume_momentum, "
            "mean_reversion, volatility_breakout. Set params.kind to disambiguate."
        )

    for definition in feature_definitions:
        if not isinstance(definition, dict):
            continue
        feature_type = definition.get("type")
        params = dict(definition.get("params") or {})

        if feature_type == "technical":
            price_features.extend(
                [
                    "close_to_open",
                    "high_to_low",
                    "close_to_high",
                    "close_to_low",
                    "price_range",
                ]
            )
            if "period" in params:
                momentum_features["rsi_period"] = int(params["period"])
        elif feature_type == "custom":
            custom_name = str(definition.get("name") or "custom_feature")
            custom_key = _resolve_custom_feature_key(custom_name, params)
            if custom_key == "volatility_breakout":
                breakout_params = {k: v for k, v in params.items() if k != "kind"}
                if "lookback" not in breakout_params:
                    breakout_params["lookback"] = _coerce_periods(params)
                custom_features.append({"volatility_breakout": breakout_params})
                continue

            custom_features.append({custom_key: {"periods": _coerce_periods(params)}})

    features_runtime_cfg = {
        "pipeline": {
            "steps": [
                *feature_steps,
                "generate_volume_features",
                "generate_custom_features",
                "handle_missing_values",
                "remove_outliers",
                "scale_features",
                "select_features",
            ]
        },
        "price_features": price_features,
        "volume_features": [{"on_balance_volume": True}],
        "volatility_features": [{"atr_periods": [14]}],
        "custom_features": custom_features,
        "feature_combinations": [],
        "sentiment": {"enabled": False},
        "feature_selection": {"method": "recursive", "n_features": 20},
        "preprocessing": {
            "scaling": {"method": "standard", "target_range": [-1, 1]},
            "outliers": {"method": "winsorize", "limits": [0.01, 0.99]},
        },
    }
    if momentum_features:
        features_runtime_cfg["momentum_features"] = momentum_features

    return model_cfg, features_runtime_cfg


@research_app.command("run")
def cmd_research_run(
    config: str = typer.Option(
        "config/project.yaml", "-c", "--config", help="Path to project config YAML"
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip data-quality validation before training",
    ),
):
    """Run the Stage 1 canonical research workflow from project.yaml."""

    project_path = Path(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "project_name": None,
        "project_profile": None,
        "symbols": [],
        "feature_count": 0,
        "model_family": None,
        "timestamps": {"started_at": datetime.now(timezone.utc).isoformat()},
        "status": "failed",
        "run_dir": str(run_dir),
        "artifacts": {},
    }

    try:
        validation = validate_project_config(
            config_path=project_path, output_dir=run_dir
        )
        resolved_path = Path(validation["artifacts"]["resolved_config"])
        summary["artifacts"]["resolved_project_config"] = str(resolved_path)

        with resolved_path.open("r", encoding="utf-8") as handle:
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

        model_cfg, features_cfg = _project_to_runtime_configs(resolved_project)
        runtime_model_path = run_dir / "runtime_model_config.yaml"
        runtime_features_path = run_dir / "runtime_features_config.yaml"
        runtime_model_path.write_text(
            yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8"
        )
        runtime_features_path.write_text(
            yaml.safe_dump(features_cfg, sort_keys=False), encoding="utf-8"
        )

        pipeline_output = run_pipeline(
            str(runtime_model_path),
            skip_validation=skip_validation,
            include_metadata=True,
            features_config_path=str(runtime_features_path),
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

        summary["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()

        metrics_payload = {
            "status": "available" if results else "placeholder",
            "metrics_by_symbol": {
                symbol: details.get("test_metrics", {})
                for symbol, details in results.items()
                if isinstance(details, dict)
            },
        }
        if not metrics_payload["metrics_by_symbol"]:
            metrics_payload["status"] = "placeholder"
            metrics_payload["message"] = (
                "No test metrics were produced by the research pipeline."
            )

    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()
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
