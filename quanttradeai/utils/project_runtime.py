"""Helpers for mapping project config into runtime configs."""

from __future__ import annotations

from typing import Any


def project_to_runtime_configs(
    project_config: dict[str, Any],
    *,
    require_research: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_cfg = dict(project_config.get("data") or {})
    research_cfg = dict(project_config.get("research") or {})
    features_cfg = dict(project_config.get("features") or {})

    def _or_default(value: Any, default: Any) -> Any:
        return default if value is None else value

    if require_research and not research_cfg.get("enabled", True):
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
