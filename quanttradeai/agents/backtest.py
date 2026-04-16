"""Agent backtesting workflow."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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
    _validate_or_raise,
    time_aware_split,
)
from quanttradeai.models.classifier import MomentumClassifier
from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.utils.config_validator import validate_project_config
from quanttradeai.utils.project_config import compile_research_runtime_configs
from quanttradeai.utils.project_paths import resolve_project_path
from quanttradeai.utils.run_records import apply_required_run_fields, create_run_dir

from .base import AgentSimulationState, action_to_target, signal_to_action
from .context import (
    attach_prompt_context_history_columns,
    build_context_payload,
    load_agent_notes_payload,
    strip_prompt_context_history_columns,
)
from .factory import build_strategy

logger = logging.getLogger(__name__)

SUPPORTED_AGENT_KINDS = {"rule", "llm", "hybrid"}
PROMPT_SAMPLE_LIMIT = 20


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


def _agent_prompt_config(agent_config: dict[str, Any], symbol: str) -> dict[str, Any]:
    runtime_agent = dict(agent_config)
    runtime_agent["_current_symbol"] = symbol
    return runtime_agent


@dataclass(slots=True)
class ModelSignalRuntime:
    """Loaded trained model used as a hybrid context source."""

    name: str
    path: Path
    runtime_model_config_path: Path
    classifier: Any = None
    feature_preprocessor: Any = None

    def __post_init__(self) -> None:
        self.classifier = MomentumClassifier(str(self.runtime_model_config_path))
        self.classifier.load_model(str(self.path))
        self.feature_preprocessor = _load_feature_preprocessor(self.path)

    def predict(self, features_frame: pd.DataFrame) -> int:
        working = features_frame.copy()
        if self.feature_preprocessor is not None:
            required_columns = (
                getattr(self.feature_preprocessor, "feature_columns", []) or []
            )
            missing = [column for column in required_columns if column not in working]
            if missing:
                raise ValueError(
                    f"Missing required features for model signal source '{self.name}': {missing}"
                )
            working = self.feature_preprocessor.transform(working)
        feature_columns = list(self.classifier.feature_columns or [])
        missing_features = [
            column for column in feature_columns if column not in working.columns
        ]
        if missing_features:
            raise ValueError(
                f"Missing classifier features for model signal source '{self.name}': {missing_features}"
            )
        prediction = self.classifier.predict(working[feature_columns].values)
        return int(prediction[0])


def _load_model_signal_sources(
    *,
    agent_config: dict[str, Any],
    project_config_path: str | Path,
    runtime_model_config_path: Path,
) -> list[ModelSignalRuntime]:
    runtimes: list[ModelSignalRuntime] = []
    for source in agent_config.get("model_signal_sources") or []:
        if isinstance(source, str):
            raise ValueError(
                "Deprecated string model_signal_sources entries are not supported at runtime. "
                "Use objects with name and path."
            )
        source_path = resolve_project_path(project_config_path, source.get("path", ""))
        runtimes.append(
            ModelSignalRuntime(
                name=str(source.get("name")),
                path=source_path,
                runtime_model_config_path=runtime_model_config_path,
            )
        )
    return runtimes


def _build_strategy(
    *,
    agent_config: dict[str, Any],
    project_config_path: str | Path,
) -> Any:
    if agent_config.get("kind") not in SUPPORTED_AGENT_KINDS:
        raise ValueError(
            "Only rule, llm, and hybrid agents are supported by `quanttradeai agent run`."
        )
    return build_strategy(
        agent_config=agent_config,
        project_config_path=project_config_path,
    )


def run_agent_backtest(
    *,
    project_config_path: str = "config/project.yaml",
    agent_name: str,
    mode: str = "backtest",
    skip_validation: bool = False,
    project_config_override: dict[str, Any] | None = None,
    run_timestamp: str | None = None,
) -> dict[str, Any]:
    """Run a rule, LLM, or hybrid agent over the configured backtest window."""

    if mode != "backtest":
        raise ValueError(
            "Only --mode backtest is implemented for agent runs at this stage."
        )

    timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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

    try:
        validation = validate_project_config(
            config_path=project_config_path,
            output_dir=run_dir,
            project_config_override=project_config_override,
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
        if agent_config.get("kind") not in SUPPORTED_AGENT_KINDS:
            raise ValueError(
                f"Agent '{agent_name}' is kind={agent_config.get('kind')}; expected kind=rule, kind=llm, or kind=hybrid."
            )

        strategy = _build_strategy(
            agent_config=agent_config,
            project_config_path=project_config_path,
        )
        include_prompt_artifacts = agent_config.get("kind") in {"llm", "hybrid"}
        notes_payload = (
            load_agent_notes_payload(
                agent_config=agent_config,
                project_config_path=project_config_path,
            )
            if include_prompt_artifacts
            else None
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

        loader = DataLoader(str(runtime_model_path))
        processor = DataProcessor(str(runtime_features_path))
        data_dict = loader.fetch_data()
        _validate_or_raise(
            loader=loader,
            data=data_dict,
            report_path=run_dir / "validation.json",
            skip_validation=skip_validation,
        )

        model_signal_runtimes = _load_model_signal_sources(
            agent_config=agent_config,
            project_config_path=project_config_path,
            runtime_model_config_path=runtime_model_path,
        )
        feature_definitions = list(
            (project_config.get("features") or {}).get("definitions") or []
        )
        execution_cfg = dict(backtest_cfg.get("execution") or {})
        stop_loss_pct = _safe_float(model_cfg.get("trading", {}).get("stop_loss"), 0.02)
        max_risk_per_trade = _safe_float(
            (agent_config.get("risk") or {}).get("max_position_pct"),
            _safe_float(model_cfg.get("trading", {}).get("max_risk_per_trade"), 0.02),
        )
        max_portfolio_risk = _safe_float(
            model_cfg.get("trading", {}).get("max_portfolio_risk"),
            max(0.10, max_risk_per_trade),
        )
        initial_capital = _safe_float(
            model_cfg.get("trading", {}).get("initial_capital"),
            100_000.0,
        )

        prepared_data: dict[str, pd.DataFrame] = {}
        decision_records: list[dict[str, Any]] = []
        execution_records: list[dict[str, Any]] = []
        prompt_samples: list[dict[str, Any]] = []

        for symbol, df in data_dict.items():
            features_df = processor.generate_features(
                strip_prompt_context_history_columns(df)
            )
            features_df = attach_prompt_context_history_columns(
                features_df,
                source_history=df,
            )
            _, test_df, coverage = time_aware_split(features_df, model_cfg)
            symbol_dir = run_dir / "symbols" / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            _write_json(symbol_dir / "coverage.json", coverage)

            target_labels: list[int] = []
            state = AgentSimulationState()
            runtime_agent_config = _agent_prompt_config(agent_config, symbol)

            for position, (bar_timestamp, _) in enumerate(test_df.iterrows()):
                history = test_df.iloc[: position + 1]
                current_frame = history.tail(1)
                current_row = current_frame.iloc[0]
                model_signals = {
                    runtime.name: runtime.predict(current_frame)
                    for runtime in model_signal_runtimes
                }
                context = build_context_payload(
                    feature_definitions=feature_definitions,
                    agent_config=runtime_agent_config,
                    history=history,
                    current_row=current_row,
                    model_signals=model_signals,
                    state=state,
                    decision_history=decision_records,
                    execution_history=execution_records,
                    notes_payload=notes_payload,
                )
                decision = strategy.decide(
                    agent_name=agent_name,
                    symbol=symbol,
                    timestamp=bar_timestamp,
                    context=context,
                    tools=list(agent_config.get("tools") or []),
                )
                target_before = state.target_position
                target_position = action_to_target(
                    state.target_position, decision.action
                )
                target_delta = target_position - target_before
                execution_status = "simulated" if target_delta != 0 else "no_change"
                state.target_position = target_position
                state.last_action = decision.action
                state.last_reason = decision.reason
                state.decision_count += 1
                target_labels.append(target_position)

                if target_delta != 0:
                    execution_records.append(
                        {
                            "symbol": symbol,
                            "timestamp": bar_timestamp,
                            "action": decision.action,
                            "qty": abs(target_delta),
                            "price": _safe_float(current_row.get("Close"), 0.0),
                            "status": "simulated",
                        }
                    )

                decision_record = {
                    "symbol": symbol,
                    "timestamp": bar_timestamp,
                    "action": decision.action,
                    "target_position_before": target_before,
                    "target_position": target_position,
                    "target_position_after": target_position,
                    "execution_status": execution_status,
                    "reason": decision.reason,
                    "context": context,
                    "model_signals": {
                        name: {
                            "signal": signal,
                            "direction": signal_to_action(signal),
                        }
                        for name, signal in model_signals.items()
                    },
                }
                decision_records.append(decision_record)
                if (
                    include_prompt_artifacts
                    and len(prompt_samples) < PROMPT_SAMPLE_LIMIT
                ):
                    prompt_samples.append(
                        {
                            "symbol": symbol,
                            "timestamp": bar_timestamp,
                            "prompt_payload": decision.prompt_payload,
                            "response_payload": decision.response_payload,
                            "raw_response": decision.raw_response,
                        }
                    )

            prepared = test_df[
                [column for column in ("Close", "Volume") if column in test_df.columns]
            ].copy()
            if "Volume" not in prepared.columns:
                prepared["Volume"] = 1e12
            prepared["label"] = target_labels
            prepared_data[symbol] = prepared

        if not prepared_data:
            raise ValueError("Agent backtest produced no prepared symbol data.")

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
        combined_ledger_path: Path | None = None
        if combined_ledger_frames:
            combined_ledger_path = run_dir / "ledger.csv"
            pd.concat(combined_ledger_frames, ignore_index=True).to_csv(
                combined_ledger_path,
                index=False,
            )

        _write_jsonl(run_dir / "decisions.jsonl", decision_records)

        artifacts = {
            **summary["artifacts"],
            "metrics": str(run_dir / "metrics.json"),
            "equity_curve": str(run_dir / "equity_curve.csv"),
            "decisions": str(run_dir / "decisions.jsonl"),
        }
        if include_prompt_artifacts:
            _write_json(run_dir / "prompt_samples.json", prompt_samples)
            artifacts["prompt_samples"] = str(run_dir / "prompt_samples.json")
        if combined_ledger_path is not None:
            artifacts["ledger"] = str(combined_ledger_path)

        summary.update(
            {
                "status": "success",
                "agent_kind": agent_config.get("kind"),
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
            mode=mode,
            name=agent_name,
        )
        _write_json(run_dir / "summary.json", summary)
        return summary

    except Exception as exc:
        logger.error("agent_backtest_failed", exc_info=exc)
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
