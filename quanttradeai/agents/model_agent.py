"""Model-backed agent runners for canonical project workflows."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quanttradeai.backtest.backtester import compute_metrics, simulate_trades
from quanttradeai.data.loader import DataLoader
from quanttradeai.data.processor import DataProcessor
from quanttradeai.main import (
    _load_feature_preprocessor,
    _prepare_labeled_dataset,
    _validate_or_raise,
    time_aware_split,
)
from quanttradeai.models.classifier import MomentumClassifier
from quanttradeai.streaming.live_trading import LiveTradingEngine
from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.utils.config_validator import validate_project_config
from quanttradeai.utils.project_config import (
    compile_live_position_manager_runtime_config,
    compile_live_risk_runtime_config,
    compile_live_streaming_runtime_config,
    compile_paper_streaming_runtime_config,
    compile_research_runtime_configs,
)
from quanttradeai.utils.project_paths import resolve_project_path
from quanttradeai.utils.run_records import apply_required_run_fields, create_run_dir

from .base import signal_to_action

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return value
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, default=_json_default))
            handle.write("\n")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _initialize_model_agent_run(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    project_config_path: str,
    agent_name: str,
    mode: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    Path,
    Path | None,
    Path | None,
    Path | None,
]:
    validation = validate_project_config(
        config_path=project_config_path,
        output_dir=run_dir,
    )
    resolved_path = Path(validation["artifacts"]["resolved_config"])
    summary["warnings"] = list(validation.get("warnings", []))
    summary["artifacts"]["resolved_project_config"] = str(resolved_path)
    project_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    agent_config = next(
        (
            dict(agent)
            for agent in project_config.get("agents") or []
            if agent.get("name") == agent_name
        ),
        None,
    )
    if agent_config is None:
        raise ValueError(f"Agent '{agent_name}' not found in project config.")
    if agent_config.get("kind") != "model":
        raise ValueError(
            f"Agent '{agent_name}' is kind={agent_config.get('kind')}; expected kind=model."
        )

    model_cfg, features_cfg, backtest_cfg = compile_research_runtime_configs(
        project_config,
        require_research=False,
    )
    runtime_model_path = run_dir / "runtime_model_config.yaml"
    runtime_features_path = run_dir / "runtime_features_config.yaml"
    runtime_backtest_path = run_dir / "runtime_backtest_config.yaml"
    runtime_model_path.write_text(
        yaml.safe_dump(model_cfg, sort_keys=False),
        encoding="utf-8",
    )
    runtime_features_path.write_text(
        yaml.safe_dump(features_cfg, sort_keys=False),
        encoding="utf-8",
    )
    runtime_backtest_path.write_text(
        yaml.safe_dump(backtest_cfg, sort_keys=False),
        encoding="utf-8",
    )
    summary["artifacts"]["runtime_model_config"] = str(runtime_model_path)
    summary["artifacts"]["runtime_features_config"] = str(runtime_features_path)
    summary["artifacts"]["runtime_backtest_config"] = str(runtime_backtest_path)

    runtime_streaming_path: Path | None = None
    runtime_risk_path: Path | None = None
    runtime_position_manager_path: Path | None = None
    if mode in {"paper", "live"}:
        if mode == "paper":
            streaming_cfg = compile_paper_streaming_runtime_config(project_config)
        else:
            streaming_cfg = compile_live_streaming_runtime_config(project_config)
        runtime_streaming_path = run_dir / "runtime_streaming_config.yaml"
        runtime_streaming_path.write_text(
            yaml.safe_dump(streaming_cfg, sort_keys=False),
            encoding="utf-8",
        )
        summary["artifacts"]["runtime_streaming_config"] = str(runtime_streaming_path)
    if mode == "live":
        runtime_risk_cfg = compile_live_risk_runtime_config(project_config)
        runtime_risk_path = run_dir / "runtime_risk_config.yaml"
        runtime_risk_path.write_text(
            yaml.safe_dump(runtime_risk_cfg, sort_keys=False),
            encoding="utf-8",
        )
        summary["artifacts"]["runtime_risk_config"] = str(runtime_risk_path)

        runtime_position_manager_cfg = compile_live_position_manager_runtime_config(
            project_config
        )
        runtime_position_manager_path = run_dir / "runtime_position_manager_config.yaml"
        runtime_position_manager_path.write_text(
            yaml.safe_dump(runtime_position_manager_cfg, sort_keys=False),
            encoding="utf-8",
        )
        summary["artifacts"]["runtime_position_manager_config"] = str(
            runtime_position_manager_path
        )

    return (
        project_config,
        agent_config,
        model_cfg,
        runtime_model_path,
        runtime_features_path,
        runtime_backtest_path,
        runtime_streaming_path,
        runtime_risk_path,
        runtime_position_manager_path,
    )


def _start_model_agent_run(
    *, agent_name: str, mode: str
) -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir, run_id = create_run_dir(
        run_type="agent",
        mode=mode,
        name=agent_name,
        timestamp=timestamp,
    )
    summary: dict[str, Any] = {
        "agent_name": agent_name,
        "status": "failed",
        "timestamps": {"started_at": datetime.now(timezone.utc).isoformat()},
        "artifacts": {},
        "warnings": [],
        "symbols": [],
    }
    apply_required_run_fields(
        summary,
        run_dir=run_dir,
        run_type="agent",
        mode=mode,
        name=agent_name,
    )
    summary["run_id"] = run_id
    return run_dir, summary


def _resolve_risk_settings(
    agent_config: dict[str, Any],
    model_cfg: dict[str, Any],
) -> tuple[float, float, float, float]:
    trading_cfg = dict(model_cfg.get("trading") or {})
    agent_risk_cfg = dict(agent_config.get("risk") or {})
    stop_loss_pct = _safe_float(trading_cfg.get("stop_loss"), 0.02)
    initial_capital = _safe_float(trading_cfg.get("initial_capital"), 100_000.0)
    max_risk_per_trade = _safe_float(
        agent_risk_cfg.get("max_position_pct"),
        _safe_float(trading_cfg.get("max_risk_per_trade"), 0.02),
    )
    max_portfolio_risk = _safe_float(
        agent_risk_cfg.get("max_portfolio_risk"),
        _safe_float(trading_cfg.get("max_portfolio_risk"), max_risk_per_trade),
    )
    return (
        initial_capital,
        stop_loss_pct,
        max_risk_per_trade,
        max_portfolio_risk,
    )


def run_model_agent_backtest(
    *,
    project_config_path: str = "config/project.yaml",
    agent_name: str,
    skip_validation: bool = False,
) -> dict[str, Any]:
    """Run a model agent over the configured backtest window."""

    run_dir, summary = _start_model_agent_run(agent_name=agent_name, mode="backtest")

    try:
        (
            _project_config,
            agent_config,
            model_cfg,
            runtime_model_path,
            runtime_features_path,
            runtime_backtest_path,
            _runtime_streaming_path,
            _runtime_risk_path,
            _runtime_position_manager_path,
        ) = _initialize_model_agent_run(
            run_dir=run_dir,
            summary=summary,
            project_config_path=project_config_path,
            agent_name=agent_name,
            mode="backtest",
        )
        model_path = resolve_project_path(
            project_config_path,
            (agent_config.get("model") or {}).get("path", ""),
        )
        classifier = MomentumClassifier(str(runtime_model_path))
        classifier.load_model(str(model_path))
        preprocessor = _load_feature_preprocessor(model_path)
        loader = DataLoader(str(runtime_model_path))
        processor = DataProcessor(str(runtime_features_path))
        data_dict = loader.fetch_data()
        _validate_or_raise(
            loader=loader,
            data=data_dict,
            report_path=run_dir / "validation.json",
            skip_validation=skip_validation,
        )

        (
            initial_capital,
            stop_loss_pct,
            max_risk_per_trade,
            max_portfolio_risk,
        ) = _resolve_risk_settings(agent_config, model_cfg)
        backtest_cfg = yaml.safe_load(runtime_backtest_path.read_text("utf-8")) or {}
        execution_cfg = dict(backtest_cfg.get("execution") or {})

        prepared_data: dict[str, pd.DataFrame] = {}
        decision_records: list[dict[str, Any]] = []

        for symbol, df in data_dict.items():
            dataset = _prepare_labeled_dataset(
                processor,
                df,
                model_cfg,
                preprocessor=preprocessor,
            )
            _, test_df, coverage = time_aware_split(dataset, model_cfg)
            symbol_dir = run_dir / "symbols" / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            _write_json(symbol_dir / "coverage.json", coverage)

            feature_columns = list(classifier.feature_columns or [])
            missing_columns = [
                column for column in feature_columns if column not in test_df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing classifier features for model agent '{agent_name}' on {symbol}: {missing_columns}"
                )

            predictions = classifier.predict(test_df[feature_columns].values)
            prepared = test_df[
                [column for column in ("Close", "Volume") if column in test_df.columns]
            ].copy()
            if "Volume" not in prepared.columns:
                prepared["Volume"] = 1e12
            prepared["label"] = predictions
            prepared_data[symbol] = prepared

            for bar_timestamp, signal in zip(test_df.index, predictions):
                signal_value = int(signal)
                decision_records.append(
                    {
                        "symbol": symbol,
                        "timestamp": bar_timestamp,
                        "action": signal_to_action(signal_value),
                        "target_position": signal_value,
                        "signal": signal_value,
                        "source": "model",
                        "model_path": str(model_path),
                    }
                )

        if not prepared_data:
            raise ValueError("Model agent backtest produced no prepared symbol data.")

        portfolio = PortfolioManager(
            capital=initial_capital,
            max_risk_per_trade=max_risk_per_trade,
            max_portfolio_risk=max_portfolio_risk,
        )
        results = simulate_trades(
            prepared_data,
            stop_loss_pct=stop_loss_pct,
            execution=execution_cfg,
            portfolio=portfolio,
        )

        metrics_by_symbol: dict[str, Any] = {}
        combined_ledger_frames: list[pd.DataFrame] = []
        for symbol, result_df in results.items():
            symbol_dir = run_dir / "symbols" / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            metrics = compute_metrics(result_df)
            metrics_by_symbol[symbol] = metrics
            _write_json(symbol_dir / "metrics.json", metrics)
            result_df[
                [
                    column
                    for column in ("strategy_return", "equity_curve")
                    if column in result_df
                ]
            ].to_csv(symbol_dir / "equity_curve.csv", index=True)
            ledger = result_df.attrs.get("ledger")
            if ledger is not None and not ledger.empty:
                ledger = ledger.copy()
                ledger["symbol"] = symbol
                ledger.to_csv(symbol_dir / "ledger.csv", index=False)
                combined_ledger_frames.append(ledger)

        aggregate_key = "portfolio" if "portfolio" in results else next(iter(results))
        aggregate_df = results[aggregate_key]
        aggregate_metrics = metrics_by_symbol[aggregate_key]
        aggregate_df[
            [
                column
                for column in ("strategy_return", "equity_curve")
                if column in aggregate_df
            ]
        ].to_csv(run_dir / "equity_curve.csv", index=True)
        _write_json(run_dir / "metrics.json", aggregate_metrics)
        _write_jsonl(run_dir / "decisions.jsonl", decision_records)

        combined_ledger_path: Path | None = None
        if combined_ledger_frames:
            combined_ledger_path = run_dir / "ledger.csv"
            pd.concat(combined_ledger_frames, ignore_index=True).to_csv(
                combined_ledger_path,
                index=False,
            )

        artifacts = {
            **summary["artifacts"],
            "metrics": str(run_dir / "metrics.json"),
            "equity_curve": str(run_dir / "equity_curve.csv"),
            "decisions": str(run_dir / "decisions.jsonl"),
        }
        if combined_ledger_path is not None:
            artifacts["ledger"] = str(combined_ledger_path)

        summary.update(
            {
                "status": "success",
                "agent_kind": "model",
                "symbols": sorted(prepared_data),
                "decision_count": len(decision_records),
                "metrics_by_symbol": metrics_by_symbol,
                "artifacts": artifacts,
            }
        )
        summary["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        apply_required_run_fields(
            summary,
            run_dir=run_dir,
            run_type="agent",
            mode="backtest",
            name=agent_name,
        )
        _write_json(run_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        logger.error("model_agent_backtest_failed", exc_info=exc)
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        apply_required_run_fields(
            summary,
            run_dir=run_dir,
            run_type="agent",
            mode="backtest",
            name=agent_name,
        )
        _write_json(run_dir / "summary.json", summary)
        raise


def _streaming_metrics(engine: LiveTradingEngine) -> dict[str, Any]:
    positions: dict[str, Any] = {}
    unrealized_pnl = 0.0
    for symbol, position in engine.portfolio.positions.items():
        qty = int(position.get("qty", 0))
        current_price = _safe_float(position.get("price"), 0.0)
        entry_price = _safe_float(position.get("entry_price"), current_price)
        market_value = qty * current_price
        position_unrealized = qty * (current_price - entry_price)
        unrealized_pnl += position_unrealized
        positions[symbol] = {
            "qty": qty,
            "entry_price": entry_price,
            "current_price": current_price,
            "market_value": market_value,
            "unrealized_pnl": position_unrealized,
            "stop_loss_pct": _safe_float(position.get("stop_loss_pct"), 0.0),
        }

    portfolio_value = engine.portfolio.portfolio_value
    payload = {
        "status": "available",
        "execution_count": len(engine.execution_log),
        "realized_pnl": _safe_float(engine.portfolio.realized_pnl, 0.0),
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": portfolio_value - engine.portfolio.initial_capital,
        "cash": engine.portfolio.cash,
        "portfolio_value": portfolio_value,
        "open_positions": positions,
    }
    risk_manager = getattr(engine, "risk_manager", None)
    if risk_manager is not None:
        try:
            payload["risk_metrics"] = dict(risk_manager.get_risk_metrics() or {})
        except Exception:  # pragma: no cover - defensive
            payload["risk_metrics"] = {}
        drawdown_guard = getattr(risk_manager, "drawdown_guard", None)
        if drawdown_guard is not None:
            try:
                payload["risk_status"] = drawdown_guard.check_drawdown_limits()
            except Exception:  # pragma: no cover - defensive
                pass
    return payload


def _model_decision_records_from_engine(
    engine: LiveTradingEngine,
) -> list[dict[str, Any]]:
    decision_log = getattr(engine, "decision_log", None)
    if isinstance(decision_log, list) and decision_log:
        return list(decision_log)

    decisions: list[dict[str, Any]] = []
    for execution in list(getattr(engine, "execution_log", [])):
        signal_value = int(execution.get("signal", 0))
        decisions.append(
            {
                "symbol": execution.get("symbol"),
                "timestamp": execution.get("timestamp"),
                "action": signal_to_action(signal_value),
                "target_position": signal_value,
                "signal": signal_value,
                "source": "model",
            }
        )
    return decisions


def run_model_agent_paper(
    *,
    project_config_path: str = "config/project.yaml",
    agent_name: str,
) -> dict[str, Any]:
    """Run a model agent in paper mode using the live trading engine."""

    return _run_model_agent_streaming(
        project_config_path=project_config_path,
        agent_name=agent_name,
        mode="paper",
    )


def run_model_agent_live(
    *,
    project_config_path: str = "config/project.yaml",
    agent_name: str,
) -> dict[str, Any]:
    """Run a model agent in live mode using the live trading engine."""

    return _run_model_agent_streaming(
        project_config_path=project_config_path,
        agent_name=agent_name,
        mode="live",
    )


def _run_model_agent_streaming(
    *,
    project_config_path: str,
    agent_name: str,
    mode: str,
) -> dict[str, Any]:
    """Run a model agent in paper or live mode using the live trading engine."""

    run_dir, summary = _start_model_agent_run(agent_name=agent_name, mode=mode)

    try:
        (
            _project_config,
            agent_config,
            model_cfg,
            runtime_model_path,
            runtime_features_path,
            _runtime_backtest_path,
            runtime_streaming_path,
            runtime_risk_path,
            runtime_position_manager_path,
        ) = _initialize_model_agent_run(
            run_dir=run_dir,
            summary=summary,
            project_config_path=project_config_path,
            agent_name=agent_name,
            mode=mode,
        )
        if runtime_streaming_path is None:
            raise ValueError(
                f"{mode.capitalize()} mode requires a compiled runtime streaming config."
            )

        model_path = resolve_project_path(
            project_config_path,
            (agent_config.get("model") or {}).get("path", ""),
        )
        (
            initial_capital,
            stop_loss_pct,
            max_risk_per_trade,
            max_portfolio_risk,
        ) = _resolve_risk_settings(agent_config, model_cfg)

        engine = LiveTradingEngine(
            model_config=str(runtime_model_path),
            model_path=str(model_path),
            features_config=str(runtime_features_path),
            streaming_config=str(runtime_streaming_path),
            risk_config=str(runtime_risk_path) if runtime_risk_path else None,
            position_manager_config=(
                str(runtime_position_manager_path)
                if runtime_position_manager_path
                else None
            ),
            initial_capital=initial_capital,
            max_risk_per_trade=max_risk_per_trade,
            max_portfolio_risk=max_portfolio_risk,
            stop_loss_pct=stop_loss_pct,
        )
        asyncio.run(engine.start())

        decision_records = _model_decision_records_from_engine(engine)
        if decision_records:
            _write_jsonl(run_dir / "decisions.jsonl", decision_records)
        _write_jsonl(run_dir / "executions.jsonl", list(engine.execution_log))
        metrics_payload = _streaming_metrics(engine)
        metrics_payload["decision_count"] = len(decision_records)
        _write_json(run_dir / "metrics.json", metrics_payload)

        artifacts = {
            **summary["artifacts"],
            "metrics": str(run_dir / "metrics.json"),
            "executions": str(run_dir / "executions.jsonl"),
        }
        if decision_records:
            artifacts["decisions"] = str(run_dir / "decisions.jsonl")
        summary.update(
            {
                "status": "success",
                "agent_kind": "model",
                "symbols": sorted(
                    set(engine._history.keys())
                    | set(
                        (yaml.safe_load(runtime_model_path.read_text("utf-8")) or {})
                        .get("data", {})
                        .get("symbols", [])
                    )
                ),
                "decision_count": len(decision_records),
                "execution_count": len(engine.execution_log),
                "artifacts": artifacts,
                "metrics": metrics_payload,
            }
        )
        summary["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        apply_required_run_fields(
            summary,
            run_dir=run_dir,
            run_type="agent",
            mode=mode,
            name=agent_name,
        )
        _write_json(run_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        logger.error("model_agent_paper_failed", exc_info=exc)
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        apply_required_run_fields(
            summary,
            run_dir=run_dir,
            run_type="agent",
            mode=mode,
            name=agent_name,
        )
        _write_json(run_dir / "summary.json", summary)
        raise
