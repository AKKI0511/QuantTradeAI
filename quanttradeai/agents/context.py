"""Deterministic agent prompt context assembly."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import AgentSimulationState, target_position_label


def _serialize_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return value
    return value


def _infer_feature_columns(
    definition: dict[str, Any],
    available_columns: set[str],
) -> list[str]:
    name = str(definition.get("name") or "")
    feature_type = definition.get("type")
    params = dict(definition.get("params") or {})

    if name in available_columns:
        return [name]

    if feature_type == "technical":
        lowered = name.lower()
        if lowered.startswith("rsi") and "rsi" in available_columns:
            return ["rsi"]
        if lowered.startswith("macd"):
            return [
                column
                for column in ("macd", "macd_signal", "macd_hist")
                if column in available_columns
            ]

    if feature_type == "custom":
        periods = (
            params.get("periods")
            or params.get("lookback")
            or params.get("period")
            or params.get("window")
        )
        if periods is None:
            periods = []
        if not isinstance(periods, list):
            periods = [periods]
        kind = params.get("kind")
        lowered = name.lower()
        if kind is None:
            if lowered.startswith("price_"):
                kind = "price_momentum"
            elif lowered.startswith("volume_"):
                kind = "volume_momentum"
            elif "reversion" in lowered:
                kind = "mean_reversion"
            elif "breakout" in lowered:
                kind = "volatility_breakout"
        if kind in {
            "price_momentum",
            "volume_momentum",
            "mean_reversion",
            "volatility_breakout",
        }:
            inferred = [
                f"{kind}_{int(period)}"
                for period in periods
                if f"{kind}_{int(period)}" in available_columns
            ]
            if inferred:
                return inferred
            if kind in available_columns:
                return [kind]

    return []


def build_context_payload(
    *,
    feature_definitions: list[dict[str, Any]],
    agent_config: dict[str, Any],
    history: pd.DataFrame,
    current_row: pd.Series,
    model_signals: dict[str, int],
    state: AgentSimulationState,
) -> dict[str, Any]:
    """Assemble deterministic prompt context for the current bar."""

    context_cfg = dict(agent_config.get("context") or {})
    available_columns = set(history.columns)
    definition_map = {
        str(item.get("name")): item
        for item in feature_definitions
        if isinstance(item, dict) and item.get("name")
    }

    payload: dict[str, Any] = {
        "agent": {
            "name": agent_config.get("name"),
            "kind": agent_config.get("kind"),
        },
        "symbol": agent_config.get("_current_symbol"),
        "timestamp": _serialize_scalar(current_row.name),
        "available_tools": list(agent_config.get("tools") or []),
    }

    market_data_cfg = context_cfg.get("market_data")
    if market_data_cfg:
        resolved_market_data = (
            market_data_cfg if isinstance(market_data_cfg, dict) else {"enabled": True}
        )
        if resolved_market_data.get("enabled", True):
            lookback_bars = int(resolved_market_data.get("lookback_bars", 20))
            recent = history.tail(lookback_bars)
            payload["market_data"] = {
                "timeframe": resolved_market_data.get("timeframe", "1d"),
                "bars": [
                    {
                        "timestamp": _serialize_scalar(index),
                        "open": _serialize_scalar(row.get("Open")),
                        "high": _serialize_scalar(row.get("High")),
                        "low": _serialize_scalar(row.get("Low")),
                        "close": _serialize_scalar(row.get("Close")),
                        "volume": _serialize_scalar(row.get("Volume")),
                    }
                    for index, row in recent.iterrows()
                ],
            }

    requested_features = list(context_cfg.get("features") or [])
    if requested_features:
        payload["features"] = {}
        for feature_name in requested_features:
            definition = definition_map.get(feature_name, {"name": feature_name})
            columns = _infer_feature_columns(definition, available_columns)
            if not columns and feature_name in available_columns:
                columns = [feature_name]
            payload["features"][feature_name] = {
                column: _serialize_scalar(current_row.get(column))
                for column in columns
                if column in current_row.index
            }

    requested_signals = list(context_cfg.get("model_signals") or [])
    if requested_signals:
        payload["model_signals"] = {
            signal_name: {
                "signal": model_signals.get(signal_name),
                "direction": (
                    "buy"
                    if (model_signals.get(signal_name) or 0) > 0
                    else "sell" if (model_signals.get(signal_name) or 0) < 0 else "hold"
                ),
            }
            for signal_name in requested_signals
        }

    if context_cfg.get("positions"):
        payload["positions"] = {
            "target_position": state.target_position,
            "direction": target_position_label(state.target_position),
            "last_action": state.last_action,
        }

    if context_cfg.get("risk_state"):
        payload["risk_state"] = {
            "decision_count": state.decision_count,
            "current_direction": target_position_label(state.target_position),
            "risk_limits": dict(agent_config.get("risk") or {}),
        }

    return payload
