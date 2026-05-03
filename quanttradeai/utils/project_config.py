"""Canonical project config loading, migration, and runtime compilation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from quanttradeai.utils.config_schemas import (
    PositionManagerConfig,
    RiskManagementConfig,
)
from quanttradeai.streaming.history import ReplayWindow, parse_iso_date


def paper_replay_enabled(project_config: dict[str, Any]) -> bool:
    streaming_cfg = dict((project_config.get("data") or {}).get("streaming") or {})
    replay_cfg = dict(streaming_cfg.get("replay") or {})
    return bool(replay_cfg.get("enabled", False))


def resolve_paper_replay_window(project_config: dict[str, Any]) -> ReplayWindow | None:
    if not paper_replay_enabled(project_config):
        return None

    data_cfg = dict(project_config.get("data") or {})
    streaming_cfg = dict(data_cfg.get("streaming") or {})
    replay_cfg = dict(streaming_cfg.get("replay") or {})

    start_date = (
        replay_cfg.get("start_date")
        or data_cfg.get("test_start")
        or data_cfg.get("start_date")
    )
    end_date = (
        replay_cfg.get("end_date")
        or data_cfg.get("test_end")
        or data_cfg.get("end_date")
    )
    if not start_date or not end_date:
        raise ValueError(
            "Replay-enabled paper mode requires replay dates or a data/test window in config/project.yaml."
        )

    replay_start = parse_iso_date(
        str(start_date), field_name="data.streaming.replay.start_date"
    )
    replay_end = parse_iso_date(
        str(end_date), field_name="data.streaming.replay.end_date"
    )
    data_start = parse_iso_date(
        str(data_cfg.get("start_date")), field_name="data.start_date"
    )
    data_end = parse_iso_date(str(data_cfg.get("end_date")), field_name="data.end_date")
    if replay_start < data_start or replay_end > data_end:
        raise ValueError(
            "data.streaming.replay window must stay within data.start_date and data.end_date."
        )
    if replay_start > replay_end:
        raise ValueError(
            "data.streaming.replay.start_date must be on or before data.streaming.replay.end_date."
        )

    return ReplayWindow(
        start_date=replay_start.isoformat(),
        end_date=replay_end.isoformat(),
        pace_delay_ms=int(replay_cfg.get("pace_delay_ms", 0) or 0),
    )


@dataclass
class LoadedProjectConfig:
    raw: dict[str, Any]
    source_path: str
    warnings: list[str]


def extract_canonical_live_risk_config(
    project_config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return canonical live risk config, accepting nested legacy fallback."""

    top_level_risk = dict(project_config.get("risk") or {})
    position_manager_cfg = dict(project_config.get("position_manager") or {})
    nested_risk = dict(position_manager_cfg.get("risk_management") or {})

    if top_level_risk:
        return top_level_risk, False
    if nested_risk:
        return nested_risk, True
    return {}, False


def normalize_live_risk_compatibility(
    project_config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Normalize legacy nested live-risk settings into canonical top-level risk."""

    normalized = dict(project_config)
    warnings: list[str] = []
    risk_cfg, used_nested_fallback = extract_canonical_live_risk_config(normalized)
    if used_nested_fallback:
        normalized["risk"] = risk_cfg
        warnings.append(
            "position_manager.risk_management is legacy compatibility only; it was treated as the canonical top-level risk section during validation."
        )
    return normalized, warnings


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


def _as_int_list(raw_value: Any) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [int(value) for value in raw_value]
    return [int(raw_value)]


def _sma_period_from_feature_name(feature_name: str) -> int | None:
    normalized_name = str(feature_name or "").strip().lower()
    if not normalized_name.startswith("sma_"):
        return None
    suffix = normalized_name.removeprefix("sma_")
    if not suffix.isdigit():
        return None
    period = int(suffix)
    return period if period > 0 else None


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


def _strip_position_manager_risk_management(
    position_manager_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(position_manager_cfg, dict):
        return {}
    sanitized = dict(position_manager_cfg)
    sanitized.pop("risk_management", None)
    return sanitized


def load_project_config(
    config_path: Path | str = "config/project.yaml",
) -> LoadedProjectConfig:
    path = Path(config_path)
    raw = _load_yaml_mapping(path)
    normalized_raw, warnings = normalize_live_risk_compatibility(raw)
    return LoadedProjectConfig(
        raw=normalized_raw,
        source_path=str(path),
        warnings=warnings,
    )


def compile_research_runtime_configs(
    project_config: dict[str, Any],
    *,
    require_research: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compile canonical project config into runtime research configs."""

    data_cfg = dict(project_config.get("data") or {})
    research_cfg = dict(project_config.get("research") or {})
    features_cfg = dict(project_config.get("features") or {})
    compat_news_cfg = dict(project_config.get("news") or {})
    compat_training_cfg = dict(project_config.get("training") or {})
    compat_trading_cfg = dict(project_config.get("trading") or {})
    compat_execution_cfg = dict(project_config.get("execution") or {})

    if require_research and not research_cfg.get("enabled", True):
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
    sma_periods: set[int] = set()

    for definition in technical_definitions:
        params = dict(definition.get("params") or {})
        feature_name = str(definition.get("name") or "")
        price_features.extend(params.get("price_features") or [])

        named_sma_period = _sma_period_from_feature_name(feature_name)
        if named_sma_period is not None:
            sma_periods.add(named_sma_period)
        sma_periods.update(_as_int_list(params.get("sma_periods")))
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

    runtime_price_features: list[str] | dict[str, Any] = price_features
    if sma_periods:
        runtime_price_features = {name: True for name in price_features}
        runtime_price_features["sma_periods"] = sorted(sma_periods)

    runtime_features_cfg: dict[str, Any] = {
        "pipeline": {"steps": steps},
        "price_features": runtime_price_features,
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


def compile_streaming_runtime_config(
    project_config: dict[str, Any],
    *,
    mode: str = "paper",
    require_realtime: bool = False,
) -> dict[str, Any]:
    """Compile canonical project streaming settings into gateway runtime YAML."""

    data_cfg = dict(project_config.get("data") or {})
    streaming_cfg = dict(data_cfg.get("streaming") or {})
    if not streaming_cfg.get("enabled", False):
        raise ValueError(
            f"data.streaming.enabled must be true for `quanttradeai agent run --mode {mode}`."
        )

    symbols = list(streaming_cfg.get("symbols") or data_cfg.get("symbols") or [])
    channels = list(streaming_cfg.get("channels") or [])
    runtime_cfg: dict[str, Any] = {
        "streaming": {
            "symbols": symbols,
            "buffer_size": int(streaming_cfg.get("buffer_size", 1000)),
            "reconnect_attempts": int(streaming_cfg.get("reconnect_attempts", 5)),
        }
    }
    replay_window = (
        resolve_paper_replay_window(project_config) if mode == "paper" else None
    )
    realtime_required = require_realtime or mode == "live" or replay_window is None
    if realtime_required:
        provider = str(streaming_cfg.get("provider") or "").strip()
        websocket_url = str(streaming_cfg.get("websocket_url") or "").strip()
        if not provider:
            raise ValueError(
                f"data.streaming.provider must be configured for `quanttradeai agent run --mode {mode}`."
            )
        if not websocket_url:
            raise ValueError(
                f"data.streaming.websocket_url must be configured for `quanttradeai agent run --mode {mode}`."
            )
        if not channels:
            raise ValueError(
                f"data.streaming.channels must be configured for `quanttradeai agent run --mode {mode}`."
            )

        provider_cfg: dict[str, Any] = {
            "name": provider,
            "websocket_url": websocket_url,
            "auth_method": streaming_cfg.get("auth_method", "api_key"),
            "subscriptions": channels,
            "symbols": symbols,
        }
        if streaming_cfg.get("rate_limit"):
            provider_cfg["rate_limit"] = dict(streaming_cfg.get("rate_limit") or {})
        if streaming_cfg.get("circuit_breaker"):
            provider_cfg["circuit_breaker"] = dict(
                streaming_cfg.get("circuit_breaker") or {}
            )
        runtime_cfg["streaming"]["providers"] = [provider_cfg]
    if replay_window is not None and mode == "paper":
        runtime_cfg["streaming"]["replay"] = {
            "enabled": True,
            "start_date": replay_window.start_date,
            "end_date": replay_window.end_date,
            "pace_delay_ms": replay_window.pace_delay_ms,
        }
    health_check_interval = streaming_cfg.get("health_check_interval")
    if health_check_interval is not None:
        runtime_cfg["streaming"]["health_check_interval"] = int(health_check_interval)

    health_sections = {}
    for key in ("monitoring", "thresholds", "alerts", "metrics", "api"):
        value = streaming_cfg.get(key)
        if value:
            health_sections[key] = dict(value)
    if health_sections:
        runtime_cfg["streaming_health"] = health_sections

    return runtime_cfg


def compile_paper_streaming_runtime_config(
    project_config: dict[str, Any],
    *,
    require_realtime: bool = False,
) -> dict[str, Any]:
    return compile_streaming_runtime_config(
        project_config,
        mode="paper",
        require_realtime=require_realtime,
    )


def compile_live_streaming_runtime_config(
    project_config: dict[str, Any],
) -> dict[str, Any]:
    return compile_streaming_runtime_config(
        project_config,
        mode="live",
        require_realtime=True,
    )


def compile_live_risk_runtime_config(
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Compile canonical top-level live risk settings into runtime YAML."""

    risk_cfg, _used_nested_fallback = extract_canonical_live_risk_config(project_config)
    if not risk_cfg:
        raise ValueError("risk is required for `quanttradeai agent run --mode live`.")
    RiskManagementConfig(**risk_cfg)
    return {"risk_management": risk_cfg}


def compile_live_position_manager_runtime_config(
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Compile canonical live position-manager settings into runtime YAML."""

    position_manager_cfg = dict(project_config.get("position_manager") or {})
    risk_cfg, _used_nested_fallback = extract_canonical_live_risk_config(project_config)
    if not position_manager_cfg:
        raise ValueError(
            "position_manager is required for `quanttradeai agent run --mode live`."
        )
    if not risk_cfg:
        raise ValueError("risk is required for `quanttradeai agent run --mode live`.")

    runtime_cfg = _strip_position_manager_risk_management(position_manager_cfg)
    runtime_cfg["risk_management"] = risk_cfg
    runtime_cfg["mode"] = "live"
    PositionManagerConfig(**runtime_cfg)
    return {"position_manager": runtime_cfg}
