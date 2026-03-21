"""Canonical project config loading, migration, and runtime compilation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


LEGACY_CONFIG_FILES = {
    "model_config": "model_config.yaml",
    "features_config": "features_config.yaml",
    "backtest_config": "backtest_config.yaml",
    "risk_config": "risk_config.yaml",
    "streaming_config": "streaming.yaml",
    "position_manager_config": "position_manager.yaml",
}


DEFAULT_PROFILES = {
    "research": {"mode": "research"},
    "paper": {"mode": "paper"},
    "live": {"mode": "live"},
}


@dataclass
class LoadedProjectConfig:
    raw: dict[str, Any]
    source: str
    source_path: str
    warnings: list[str]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping at root: {path}")
    return payload


def _load_optional_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_yaml_mapping(path)


def _default_project_name(config_dir: Path) -> str:
    candidate = config_dir.resolve().parent.name.strip()
    return candidate.lower().replace(" ", "_") if candidate else "migrated_project"


def _as_int_list(raw_value: Any) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [int(value) for value in raw_value]
    return [int(raw_value)]


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


def import_legacy_project_config(
    config_dir: Path | str = "config",
) -> tuple[dict[str, Any], list[str]]:
    """Synthesize a canonical project config from legacy YAML files."""

    config_root = Path(config_dir)
    model_cfg = _load_yaml_mapping(config_root / LEGACY_CONFIG_FILES["model_config"])
    features_cfg = _load_optional_yaml(
        config_root / LEGACY_CONFIG_FILES["features_config"]
    )
    backtest_cfg = _load_optional_yaml(
        config_root / LEGACY_CONFIG_FILES["backtest_config"]
    )
    risk_cfg = _load_optional_yaml(config_root / LEGACY_CONFIG_FILES["risk_config"])
    streaming_cfg = _load_optional_yaml(
        config_root / LEGACY_CONFIG_FILES["streaming_config"]
    )
    position_manager_cfg = _load_optional_yaml(
        config_root / LEGACY_CONFIG_FILES["position_manager_config"]
    )

    warnings: list[str] = [
        "Using legacy config import. Review the migrated project config and commit it as config/project.yaml for the canonical happy path."
    ]

    feature_definitions: list[dict[str, Any]] = []
    legacy_price_features = features_cfg.get("price_features") or []
    legacy_momentum = features_cfg.get("momentum_features") or {}
    legacy_volatility = features_cfg.get("volatility_features") or []
    legacy_custom = features_cfg.get("custom_features") or []
    legacy_sentiment = features_cfg.get("sentiment") or {}

    if features_cfg:
        technical_params: dict[str, Any] = {}
        if legacy_price_features:
            technical_params["price_features"] = list(legacy_price_features)
        if isinstance(legacy_momentum, dict):
            if "rsi_period" in legacy_momentum:
                technical_params["rsi_period"] = legacy_momentum["rsi_period"]
            if "macd_params" in legacy_momentum:
                technical_params["macd_params"] = legacy_momentum["macd_params"]
            if "stoch_params" in legacy_momentum:
                technical_params["stoch_params"] = legacy_momentum["stoch_params"]
        for entry in legacy_volatility if isinstance(legacy_volatility, list) else []:
            if not isinstance(entry, dict):
                continue
            if "atr_periods" in entry:
                technical_params["atr_periods"] = entry["atr_periods"]
            if "bollinger_bands" in entry:
                technical_params["bollinger_bands"] = entry["bollinger_bands"]
            if "keltner_channels" in entry:
                technical_params["keltner_channels"] = entry["keltner_channels"]

        if technical_params:
            feature_definitions.append(
                {
                    "name": "technical_core",
                    "type": "technical",
                    "params": technical_params,
                }
            )

        for entry in legacy_custom if isinstance(legacy_custom, list) else []:
            if not isinstance(entry, dict):
                continue
            for key, value in entry.items():
                params = {"kind": key}
                if isinstance(value, dict):
                    params.update(value)
                elif isinstance(value, list):
                    params["periods"] = value
                feature_definitions.append(
                    {
                        "name": key,
                        "type": "custom",
                        "params": params,
                    }
                )

    if legacy_sentiment.get("enabled"):
        warnings.append(
            "Legacy sentiment settings were preserved as compatibility data but are not migrated into canonical research feature definitions."
        )

    data_cfg = dict(model_cfg.get("data") or {})
    research_backtest_costs = {"enabled": False, "bps": 0.0}
    transaction_costs = (backtest_cfg.get("execution") or {}).get(
        "transaction_costs"
    ) or {}
    if (
        transaction_costs.get("enabled")
        and transaction_costs.get("mode", "bps") == "bps"
    ):
        research_backtest_costs = {
            "enabled": True,
            "bps": float(transaction_costs.get("value", 0.0)),
        }
    else:
        legacy_tx_cost = float(
            (model_cfg.get("trading") or {}).get("transaction_cost", 0.0)
        )
        if legacy_tx_cost > 0:
            research_backtest_costs = {
                "enabled": True,
                "bps": legacy_tx_cost * 10000.0,
            }
        elif transaction_costs.get("enabled"):
            warnings.append(
                "Legacy backtest transaction costs were enabled in a non-bps mode and were not migrated into research.backtest.costs."
            )

    project_cfg: dict[str, Any] = {
        "project": {
            "name": _default_project_name(config_root),
            "profile": "research",
        },
        "profiles": dict(DEFAULT_PROFILES),
        "data": {
            "symbols": data_cfg.get("symbols", []),
            "start_date": data_cfg.get("start_date"),
            "end_date": data_cfg.get("end_date"),
            "timeframe": data_cfg.get("timeframe", "1d"),
            "test_start": data_cfg.get("test_start"),
            "test_end": data_cfg.get("test_end"),
            "cache_dir": data_cfg.get("cache_dir"),
            "cache_path": data_cfg.get("cache_path"),
            "cache_expiration_days": data_cfg.get("cache_expiration_days"),
            "use_cache": data_cfg.get("use_cache", True),
            "refresh": data_cfg.get("refresh", False),
            "max_workers": data_cfg.get("max_workers", 1),
        },
        "features": {"definitions": feature_definitions},
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
            "evaluation": {
                "split": "time_aware",
                "use_configured_test_window": bool(data_cfg.get("test_start")),
            },
            "backtest": {"costs": research_backtest_costs},
        },
        "agents": [],
        "deployment": {"target": "docker-compose", "mode": "paper"},
    }

    if model_cfg.get("news"):
        project_cfg["news"] = model_cfg["news"]
    if model_cfg.get("training"):
        project_cfg["training"] = model_cfg["training"]
    if model_cfg.get("trading"):
        project_cfg["trading"] = model_cfg["trading"]
    if model_cfg.get("models"):
        project_cfg["models"] = model_cfg["models"]
    if backtest_cfg.get("execution") or {}:
        project_cfg["execution"] = backtest_cfg["execution"]
    if risk_cfg:
        project_cfg["risk"] = risk_cfg.get("risk_management", risk_cfg)
    if streaming_cfg:
        project_cfg["data"]["streaming"] = streaming_cfg.get("streaming", streaming_cfg)
    if position_manager_cfg:
        project_cfg["position_manager"] = position_manager_cfg.get(
            "position_manager", position_manager_cfg
        )

    return project_cfg, warnings


def load_project_config(
    config_path: Path | str = "config/project.yaml",
    *,
    legacy_config_dir: Path | str | None = None,
) -> LoadedProjectConfig:
    if legacy_config_dir is not None:
        raw, warnings = import_legacy_project_config(legacy_config_dir)
        return LoadedProjectConfig(
            raw=raw,
            source="legacy",
            source_path=str(Path(legacy_config_dir)),
            warnings=warnings,
        )

    path = Path(config_path)
    raw = _load_yaml_mapping(path)
    return LoadedProjectConfig(
        raw=raw, source="canonical", source_path=str(path), warnings=[]
    )


def compile_research_runtime_configs(
    project_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compile canonical project config into runtime research configs."""

    data_cfg = dict(project_config.get("data") or {})
    research_cfg = dict(project_config.get("research") or {})
    features_cfg = dict(project_config.get("features") or {})
    compat_news_cfg = dict(project_config.get("news") or {})
    compat_training_cfg = dict(project_config.get("training") or {})
    compat_trading_cfg = dict(project_config.get("trading") or {})
    compat_execution_cfg = dict(project_config.get("execution") or {})

    if not research_cfg.get("enabled", True):
        raise ValueError(
            "research.enabled must be true for `quanttradeai research run`."
        )

    research_model_cfg = dict(research_cfg.get("model") or {})
    evaluation_cfg = dict(research_cfg.get("evaluation") or {})
    labels_cfg = dict(research_cfg.get("labels") or {})
    tuning_cfg = dict(research_model_cfg.get("tuning") or {})
    use_test_window = bool(evaluation_cfg.get("use_configured_test_window", True))

    def _or_default(value: Any, default: Any) -> Any:
        return default if value is None else value

    model_cfg = {
        "data": {
            "symbols": data_cfg.get("symbols", []),
            "start_date": data_cfg.get("start_date"),
            "end_date": data_cfg.get("end_date"),
            "timeframe": _or_default(data_cfg.get("timeframe"), "1d"),
            "test_start": data_cfg.get("test_start") if use_test_window else None,
            "test_end": data_cfg.get("test_end") if use_test_window else None,
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
            "enabled": bool(compat_news_cfg.get("enabled", False)),
            "provider": compat_news_cfg.get("provider", "yfinance"),
            "lookback_days": compat_news_cfg.get("lookback_days", 30),
            "symbols": compat_news_cfg.get("symbols", []),
        },
        "models": {
            "family": research_model_cfg.get("family", "voting"),
            "kind": research_model_cfg.get("kind", "classifier"),
        },
        "labels": {
            "type": labels_cfg.get("type", "forward_return"),
            "horizon": labels_cfg.get("horizon", 5),
            "buy_threshold": labels_cfg.get("buy_threshold", 0.01),
            "sell_threshold": labels_cfg.get("sell_threshold", -0.01),
        },
        "training": {
            "test_size": _or_default(compat_training_cfg.get("test_size"), 0.2),
            "random_state": _or_default(compat_training_cfg.get("random_state"), 42),
            "cv_folds": _or_default(compat_training_cfg.get("cv_folds"), 5),
            "tune_hyperparameters": bool(tuning_cfg.get("enabled", True)),
            "optuna_trials": int(_or_default(tuning_cfg.get("trials"), 50)),
        },
        "trading": {
            "initial_capital": _or_default(
                compat_trading_cfg.get("initial_capital"), 100000
            ),
            "position_size": _or_default(compat_trading_cfg.get("position_size"), 0.2),
            "stop_loss": _or_default(compat_trading_cfg.get("stop_loss"), 0.02),
            "take_profit": _or_default(compat_trading_cfg.get("take_profit"), 0.04),
            "max_positions": _or_default(compat_trading_cfg.get("max_positions"), 5),
            "transaction_cost": _or_default(
                compat_trading_cfg.get("transaction_cost"), 0.001
            ),
            "max_risk_per_trade": _or_default(
                compat_trading_cfg.get("max_risk_per_trade"), 0.02
            ),
            "max_portfolio_risk": _or_default(
                compat_trading_cfg.get("max_portfolio_risk"), 0.10
            ),
        },
    }

    definitions = features_cfg.get("definitions") or []
    technical_definitions = [
        definition
        for definition in definitions
        if isinstance(definition, dict) and definition.get("type") == "technical"
    ]
    custom_definitions = [
        definition
        for definition in definitions
        if isinstance(definition, dict) and definition.get("type") == "custom"
    ]

    steps: list[str] = []
    if technical_definitions:
        steps.append("generate_technical_indicators")
    if custom_definitions:
        steps.append("generate_custom_features")
    steps.extend(
        [
            "handle_missing_values",
            "remove_outliers",
            "scale_features",
            "select_features",
        ]
    )

    price_features: list[str] = []
    momentum_features: dict[str, Any] = {}
    volatility_features: dict[str, Any] = {
        "atr_periods": [],
        "bollinger_bands": None,
        "keltner_channels": None,
    }
    custom_features: list[dict[str, Any]] = []

    for definition in technical_definitions:
        params = dict(definition.get("params") or {})
        price_features.extend(params.get("price_features") or [])

        if "period" in params:
            momentum_features["rsi_period"] = int(params["period"])
        if "rsi_period" in params:
            momentum_features["rsi_period"] = int(params["rsi_period"])
        if "macd_params" in params:
            momentum_features["macd_params"] = params["macd_params"]
        if "stoch_params" in params:
            momentum_features["stoch_params"] = params["stoch_params"]
        if "atr_periods" in params:
            volatility_features["atr_periods"] = _as_int_list(params["atr_periods"])
        if "bollinger_bands" in params:
            volatility_features["bollinger_bands"] = params["bollinger_bands"]
        if "keltner_channels" in params:
            volatility_features["keltner_channels"] = params["keltner_channels"]

    if technical_definitions and not price_features:
        price_features = [
            "close_to_open",
            "high_to_low",
            "close_to_high",
            "close_to_low",
            "price_range",
        ]
    if technical_definitions and "rsi_period" not in momentum_features:
        momentum_features["rsi_period"] = 14

    for definition in custom_definitions:
        params = dict(definition.get("params") or {})
        custom_name = str(definition.get("name") or "custom_feature")
        custom_key = _resolve_custom_feature_key(custom_name, params)
        if custom_key == "volatility_breakout":
            breakout_params = {k: v for k, v in params.items() if k != "kind"}
            if "lookback" not in breakout_params:
                breakout_params["lookback"] = _as_int_list(
                    params.get("lookback")
                    or params.get("periods")
                    or params.get("period")
                    or params.get("window")
                )
            custom_features.append({"volatility_breakout": breakout_params})
            continue

        periods = _as_int_list(
            params.get("periods")
            or params.get("lookback")
            or params.get("period")
            or params.get("window")
        )
        custom_features.append({custom_key: {"periods": periods}})

    runtime_features_cfg: dict[str, Any] = {
        "pipeline": {"steps": steps},
        "price_features": price_features,
        "volume_features": [],
        "volatility_features": volatility_features,
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
        runtime_features_cfg["momentum_features"] = momentum_features

    execution_cfg = {
        "transaction_costs": {
            "enabled": False,
            "mode": "bps",
            "value": 0.0,
            "apply_on": "notional",
        },
        "slippage": {
            "enabled": False,
            "mode": "bps",
            "value": 0.0,
            "reference_price": "close",
        },
        "liquidity": {
            "enabled": False,
            "max_participation": 0.1,
            "volume_source": "bar_volume",
        },
        "impact": {
            "enabled": False,
            "model": "linear",
            "alpha": 0.0,
            "beta": 0.0,
            "decay": 0.0,
            "spread": 0.0,
            "average_daily_volume": 0,
        },
        "borrow_fee": {"enabled": False, "rate_bps": 0.0},
        "intrabar": {
            "enabled": False,
            "tick_column": "ticks",
            "drift": 0.0,
            "volatility": 0.0,
            "synthetic_ticks": 0,
        },
    }

    if compat_execution_cfg:
        for key, value in compat_execution_cfg.items():
            execution_cfg[key] = value

    backtest_costs_cfg = dict((research_cfg.get("backtest") or {}).get("costs") or {})
    if backtest_costs_cfg.get("enabled"):
        execution_cfg.setdefault("transaction_costs", {})
        execution_cfg["transaction_costs"].update(
            {
                "enabled": True,
                "mode": "bps",
                "value": float(backtest_costs_cfg.get("bps", 0.0)),
            }
        )

    runtime_backtest_cfg = {
        "data_path": "derived_from_research_run.csv",
        "execution": execution_cfg,
    }

    return model_cfg, runtime_features_cfg, runtime_backtest_cfg
