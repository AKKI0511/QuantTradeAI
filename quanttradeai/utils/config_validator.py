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
from typing import Callable, Dict, Iterable, Mapping

import yaml
from pydantic import ValidationError

from quanttradeai.utils.config_schemas import (
    BacktestConfigSchema,
    FeaturesConfigSchema,
    ModelConfigSchema,
    PositionManagerConfig,
    RiskManagementConfig,
)
from quanttradeai.utils.impact_loader import ImpactConfigError, load_impact_config


DEFAULT_CONFIG_PATHS: Dict[str, Path] = {
    "model_config": Path("config/model_config.yaml"),
    "features_config": Path("config/features_config.yaml"),
    "backtest_config": Path("config/backtest_config.yaml"),
    "impact_config": Path("config/impact_config.yaml"),
    "risk_config": Path("config/risk_config.yaml"),
    "streaming_config": Path("config/streaming.yaml"),
    "position_manager_config": Path("config/position_manager.yaml"),
}


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
