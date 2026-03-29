"""Centralized configuration validation utilities.

Provides a helper to validate all first-class YAML configuration files used
by QuantTradeAI and emit consolidated JSON/CSV reports. Validation reuses the
project's existing Pydantic schemas and loader helpers to mirror runtime
behavior.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Mapping

import yaml
from pydantic import ValidationError

from quanttradeai.utils.config_schemas import (
    BacktestConfigSchema,
    FeaturesConfigSchema,
    ModelConfigSchema,
    PositionManagerConfig,
    ProjectConfigSchema,
    RiskManagementConfig,
)
from quanttradeai.utils.impact_loader import ImpactConfigError, load_impact_config
from quanttradeai.utils.project_paths import resolve_project_path
from quanttradeai.utils.project_config import load_project_config


DEFAULT_CONFIG_PATHS: Dict[str, Path] = {
    "model_config": Path("config/model_config.yaml"),
    "features_config": Path("config/features_config.yaml"),
    "backtest_config": Path("config/backtest_config.yaml"),
    "impact_config": Path("config/impact_config.yaml"),
    "risk_config": Path("config/risk_config.yaml"),
    "streaming_config": Path("config/streaming.yaml"),
    "position_manager_config": Path("config/position_manager.yaml"),
}


REQUIRED_PROJECT_SECTIONS = [
    "project",
    "profiles",
    "data",
    "features",
    "research",
    "agents",
    "deployment",
]
LEGACY_PROJECT_SECTIONS = {
    "models",
    "training",
    "trading",
    "execution",
    "risk_management",
    "pipeline",
    "news",
}


def _validate_agent_project_sections(
    *,
    resolved: dict[str, Any],
    config_path: Path,
) -> list[str]:
    warnings: list[str] = []
    errors: list[str] = []

    feature_names = {
        str(item.get("name"))
        for item in (resolved.get("features") or {}).get("definitions", [])
        if isinstance(item, dict) and item.get("name")
    }
    data_streaming_cfg = dict((resolved.get("data") or {}).get("streaming") or {})

    for agent in resolved.get("agents") or []:
        agent_name = agent.get("name", "<unknown>")
        agent_kind = agent.get("kind")
        context_cfg = dict(agent.get("context") or {})
        llm_cfg = agent.get("llm") or {}
        model_cfg = agent.get("model") or {}

        if agent_kind in {"llm", "hybrid"}:
            prompt_file = llm_cfg.get("prompt_file")
            prompt_path = resolve_project_path(config_path, prompt_file or "")
            if not prompt_path.is_file():
                errors.append(
                    f"Agent '{agent_name}' prompt file does not exist: {prompt_path}"
                )
        if agent.get("mode") == "paper" and not data_streaming_cfg.get(
            "enabled", False
        ):
            errors.append(
                f"Agent '{agent_name}' is configured for paper mode but data.streaming.enabled is not true."
            )

        if agent_kind == "model":
            model_path_raw = str(model_cfg.get("path", "")).strip()
            if not model_path_raw:
                errors.append(f"Agent '{agent_name}' model path must not be empty.")
            model_path = resolve_project_path(config_path, model_path_raw)
            if model_path_raw and not model_path.exists():
                errors.append(
                    f"Agent '{agent_name}' model path does not exist: {model_path}"
                )

        missing_features = sorted(
            feature_name
            for feature_name in (context_cfg.get("features") or [])
            if feature_name not in feature_names
        )
        if missing_features:
            errors.append(
                f"Agent '{agent_name}' references unknown features: "
                + ", ".join(missing_features)
            )

        source_names: list[str] = []
        for source in agent.get("model_signal_sources") or []:
            if isinstance(source, str):
                warnings.append(
                    f"Agent '{agent_name}' uses deprecated string model_signal_sources entry "
                    f"'{source}'. Use objects with name and path."
                )
                source_names.append(source)
                continue

            source_name = source.get("name")
            source_names.append(str(source_name))
            source_path = resolve_project_path(config_path, source.get("path", ""))
            if not source_path.exists():
                errors.append(
                    f"Agent '{agent_name}' model signal source '{source_name}' does not exist: "
                    f"{source_path}"
                )

        missing_signal_refs = sorted(
            signal_name
            for signal_name in (context_cfg.get("model_signals") or [])
            if signal_name not in source_names
        )
        if missing_signal_refs:
            errors.append(
                f"Agent '{agent_name}' references unknown model signals: "
                + ", ".join(missing_signal_refs)
            )

    if errors:
        raise ValueError("\n".join(errors))

    return warnings


def _render_project_summary(resolved: dict, warnings: list[str]) -> dict:
    agents = resolved.get("agents") or []
    data = resolved.get("data") or {}
    project = resolved.get("project") or {}
    deployment = resolved.get("deployment") or {}
    return {
        "project": {
            "name": project.get("name"),
            "profile": project.get("profile"),
        },
        "data": {
            "symbols": data.get("symbols", []),
            "timeframe": data.get("timeframe"),
            "date_range": {
                "start": data.get("start_date"),
                "end": data.get("end_date"),
            },
            "test_window": {
                "start": data.get("test_start"),
                "end": data.get("test_end"),
            },
        },
        "profiles": sorted((resolved.get("profiles") or {}).keys()),
        "feature_definitions": len(
            (resolved.get("features") or {}).get("definitions", [])
        ),
        "research_enabled": bool(
            (resolved.get("research") or {}).get("enabled", False)
        ),
        "agents": [
            {
                "name": agent.get("name"),
                "kind": agent.get("kind"),
                "mode": agent.get("mode"),
            }
            for agent in agents
        ],
        "deployment": deployment,
        "warnings": warnings,
    }


def _merge_preserving_unknown(raw_value: Any, validated_value: Any) -> Any:
    """Merge validated config into raw config while preserving unknown keys.

    This keeps schema-normalized values for known fields while ensuring any
    forward-compatible/extra settings from the original file remain in the
    emitted resolved artifact.
    """

    if isinstance(raw_value, dict) and isinstance(validated_value, dict):
        merged = deepcopy(raw_value)
        for key, value in validated_value.items():
            if key in merged:
                merged[key] = _merge_preserving_unknown(merged[key], value)
            else:
                merged[key] = value
        return merged

    if isinstance(raw_value, list) and isinstance(validated_value, list):
        merged_items: list[Any] = []
        for idx, validated_item in enumerate(validated_value):
            if idx < len(raw_value):
                merged_items.append(
                    _merge_preserving_unknown(raw_value[idx], validated_item)
                )
            else:
                merged_items.append(validated_item)
        if len(raw_value) > len(validated_value):
            merged_items.extend(raw_value[len(validated_value) :])
        return merged_items

    return validated_value


def validate_project_config(
    config_path: Path | str = "config/project.yaml",
    *,
    output_dir: Path | str = "reports/config_validation",
    legacy_config_dir: Path | str | None = None,
    timestamp_subdir: bool = True,
) -> Dict:
    loaded = load_project_config(
        config_path=config_path,
        legacy_config_dir=legacy_config_dir,
    )
    raw = loaded.raw
    path = (
        Path(config_path)
        if loaded.source == "canonical"
        else Path(legacy_config_dir or Path(config_path).parent) / "project.yaml"
    )

    missing_sections = [name for name in REQUIRED_PROJECT_SECTIONS if name not in raw]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Project config missing required section(s): {missing}")

    cfg = ProjectConfigSchema(**raw)
    resolved = _merge_preserving_unknown(raw, cfg.model_dump(mode="json"))

    unused_legacy_sections = sorted(LEGACY_PROJECT_SECTIONS.intersection(raw.keys()))
    warnings = list(loaded.warnings)
    if unused_legacy_sections:
        warnings.append(
            "Legacy compatibility sections present: "
            + ", ".join(unused_legacy_sections)
            + ". They are accepted for migration compatibility but should be moved into canonical project config sections."
        )
    warnings.extend(
        _validate_agent_project_sections(
            resolved=resolved,
            config_path=path,
        )
    )

    summary = _render_project_summary(resolved=resolved, warnings=warnings)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / timestamp if timestamp_subdir else Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = run_dir / "resolved_project_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    result = {
        "timestamp": timestamp,
        "config_path": loaded.source_path,
        "all_passed": True,
        "summary": summary,
        "warnings": warnings,
        "source": loaded.source,
        "artifacts": {
            "resolved_config": str(resolved_path),
            "summary": str(summary_path),
        },
    }

    if loaded.source == "legacy":
        migrated_path = run_dir / "migrated_project_config.yaml"
        with migrated_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(resolved, f, sort_keys=False)
        result["artifacts"]["migrated_project_config"] = str(migrated_path)

    return result


@dataclass
class ValidationResult:
    """Serializable validation outcome."""

    name: str
    path: str
    passed: bool
    details: Dict | None = None
    error: str | None = None

    def to_dict(self) -> Dict:
        payload = {
            "path": self.path,
            "passed": self.passed,
        }
        if self.details:
            payload["details"] = self.details
        if self.error:
            payload["error"] = self.error
        return payload


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at root: {path}")
    return data


def _validate_model_config(path: Path) -> Dict:
    raw = _load_yaml(path)
    cfg = ModelConfigSchema(**raw)
    return {
        "symbols": list(cfg.data.symbols),
        "timeframe": cfg.data.timeframe,
        "test_window": {
            "start": cfg.data.test_start,
            "end": cfg.data.test_end,
        },
    }


def _validate_features_config(path: Path) -> Dict:
    raw = _load_yaml(path)
    cfg = FeaturesConfigSchema(**raw)
    steps: Iterable[str] = cfg.pipeline.steps if cfg.pipeline else []
    return {
        "pipeline_steps": list(steps),
        "price_features": sorted(cfg.price_features.enabled),
    }


def _validate_backtest_config(path: Path) -> Dict:
    raw = _load_yaml(path)
    cfg = BacktestConfigSchema(**raw)
    execution = cfg.execution
    return {
        "transaction_costs": execution.transaction_costs.enabled,
        "slippage": execution.slippage.enabled,
        "impact": execution.impact.enabled,
        "liquidity": execution.liquidity.enabled,
        "borrow_fee": execution.borrow_fee.enabled,
        "intrabar": execution.intrabar.enabled,
    }


def _validate_impact_config(path: Path) -> Dict:
    validated = load_impact_config(path)
    return {"asset_classes": sorted(validated)}


def _validate_risk_config(path: Path) -> Dict:
    raw = _load_yaml(path)
    cfg = RiskManagementConfig(**raw.get("risk_management", raw))
    dd_cfg = cfg.drawdown_protection
    to_cfg = cfg.turnover_limits
    return {
        "drawdown_protection_enabled": dd_cfg.enabled,
        "turnover_limits": {
            "daily_max": to_cfg.daily_max,
            "weekly_max": to_cfg.weekly_max,
            "monthly_max": to_cfg.monthly_max,
        },
    }


def _validate_streaming_config(path: Path) -> Dict:
    raw = _load_yaml(path)
    streaming_cfg = raw.get("streaming")
    if not isinstance(streaming_cfg, dict):
        raise ValueError("streaming config must include a 'streaming' mapping")
    providers = streaming_cfg.get("providers")
    provider_count = len(providers) if isinstance(providers, list) else 0
    return {
        "providers": provider_count,
        "symbols": streaming_cfg.get("symbols", []),
    }


def _validate_position_manager_config(path: Path) -> Dict:
    raw = _load_yaml(path)
    cfg = PositionManagerConfig(**raw.get("position_manager", raw))
    return {
        "mode": cfg.mode,
        "reconciliation": cfg.reconciliation,
        "risk_management": {
            "drawdown_enabled": cfg.risk_management.drawdown_protection.enabled,
        },
    }


VALIDATORS: Dict[str, Callable[[Path], Dict]] = {
    "model_config": _validate_model_config,
    "features_config": _validate_features_config,
    "backtest_config": _validate_backtest_config,
    "impact_config": _validate_impact_config,
    "risk_config": _validate_risk_config,
    "streaming_config": _validate_streaming_config,
    "position_manager_config": _validate_position_manager_config,
}


def _run_validator(name: str, path: Path) -> ValidationResult:
    validator = VALIDATORS[name]
    try:
        details = validator(path)
        return ValidationResult(name=name, path=str(path), passed=True, details=details)
    except (ValidationError, ImpactConfigError, FileNotFoundError, ValueError) as exc:
        return ValidationResult(name=name, path=str(path), passed=False, error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return ValidationResult(
            name=name, path=str(path), passed=False, error=repr(exc)
        )


def _write_reports(output_dir: Path, summary: Dict, timestamp: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"config_validation_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    csv_path = output_dir / f"config_validation_{timestamp}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "path", "passed", "error"])
        writer.writeheader()
        for name, result in summary["results"].items():
            writer.writerow(
                {
                    "name": name,
                    "path": result["path"],
                    "passed": result["passed"],
                    "error": result.get("error", ""),
                }
            )

    return {"json": str(json_path), "csv": str(csv_path)}


def validate_all(
    config_paths: Mapping[str, Path | str] | None = None,
    *,
    output_dir: Path | str = "reports/config_validation",
) -> Dict:
    """Validate all known configuration files and persist a summary report."""

    resolved_paths: Dict[str, Path] = {k: v for k, v in DEFAULT_CONFIG_PATHS.items()}
    if config_paths:
        for name, path in config_paths.items():
            if name not in VALIDATORS:
                continue
            resolved_paths[name] = Path(path)

    results: Dict[str, Dict] = {}
    for name, path in resolved_paths.items():
        result = _run_validator(name, path)
        results[name] = result.to_dict()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "all_passed": all(r["passed"] for r in results.values()),
        "results": results,
    }
    report_paths = _write_reports(Path(output_dir), summary, timestamp)
    summary["report_paths"] = report_paths
    return summary
