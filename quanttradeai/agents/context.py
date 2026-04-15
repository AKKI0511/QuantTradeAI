"""Deterministic agent prompt context assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quanttradeai.utils.project_paths import infer_project_root, resolve_project_path

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


def resolve_context_block(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        resolved = dict(value)
        resolved.setdefault("enabled", True)
        return resolved
    if value is True:
        return {"enabled": True}
    return {"enabled": False}


def context_block_enabled(value: Any) -> bool:
    return bool(resolve_context_block(value).get("enabled", False))


def resolve_agent_notes_path(
    *,
    agent_config: dict[str, Any],
    project_config_path: str | Path,
) -> Path | None:
    context_cfg = dict(agent_config.get("context") or {})
    notes_cfg = resolve_context_block(context_cfg.get("notes"))
    if not notes_cfg.get("enabled", False):
        return None

    agent_name = str(agent_config.get("name") or "agent").strip() or "agent"
    notes_file = str(notes_cfg.get("file") or f"notes/{agent_name}.md").strip()
    return resolve_project_path(project_config_path, notes_file)


def load_agent_notes_payload(
    *,
    agent_config: dict[str, Any],
    project_config_path: str | Path,
) -> dict[str, Any] | None:
    notes_path = resolve_agent_notes_path(
        agent_config=agent_config,
        project_config_path=project_config_path,
    )
    if notes_path is None:
        return None
    if not notes_path.is_file():
        raise ValueError(f"Agent notes file does not exist: {notes_path}")

    content = notes_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Agent notes file is empty: {notes_path}")

    project_root = infer_project_root(project_config_path).resolve()
    try:
        display_path = notes_path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        display_path = str(notes_path)

    return {"path": display_path, "content": content}


def _recent_execution_records(
    *,
    symbol: str | None,
    execution_history: list[dict[str, Any]] | None,
    max_entries: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in reversed(execution_history or []):
        if symbol and record.get("symbol") != symbol:
            continue
        items.append(
            {
                "timestamp": _serialize_scalar(record.get("timestamp")),
                "action": record.get("action"),
                "qty": _serialize_scalar(record.get("qty")),
                "price": _serialize_scalar(record.get("price")),
                "status": record.get("status") or record.get("execution_status"),
            }
        )
        if len(items) >= max_entries:
            break
    return items


def _recent_decision_records(
    *,
    symbol: str | None,
    decision_history: list[dict[str, Any]] | None,
    max_entries: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in reversed(decision_history or []):
        if symbol and record.get("symbol") != symbol:
            continue
        items.append(
            {
                "timestamp": _serialize_scalar(record.get("timestamp")),
                "action": record.get("action"),
                "reason": record.get("reason"),
                "execution_status": record.get("execution_status"),
                "target_position_after": _serialize_scalar(
                    record.get(
                        "target_position_after",
                        record.get("position_after", record.get("target_position")),
                    )
                ),
            }
        )
        if len(items) >= max_entries:
            break
    return items


def _recent_news_items(
    *,
    history: pd.DataFrame,
    max_items: int,
) -> list[dict[str, Any]]:
    if "text" not in history.columns:
        return []

    items: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for timestamp, value in reversed(list(history["text"].items())):
        if pd.isna(value):
            continue
        text = str(value or "").strip()
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        items.append(
            {
                "timestamp": _serialize_scalar(timestamp),
                "text": text,
            }
        )
        if len(items) >= max_items:
            break
    return items


def build_context_payload(
    *,
    feature_definitions: list[dict[str, Any]],
    agent_config: dict[str, Any],
    history: pd.DataFrame,
    current_row: pd.Series,
    model_signals: dict[str, int],
    state: AgentSimulationState,
    decision_history: list[dict[str, Any]] | None = None,
    execution_history: list[dict[str, Any]] | None = None,
    notes_payload: dict[str, Any] | None = None,
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

    if agent_config.get("kind") not in {"llm", "hybrid"}:
        return payload

    current_symbol = agent_config.get("_current_symbol")

    orders_cfg = resolve_context_block(context_cfg.get("orders"))
    if orders_cfg.get("enabled", False):
        payload["orders"] = {
            "recent_orders": _recent_execution_records(
                symbol=current_symbol,
                execution_history=execution_history,
                max_entries=int(orders_cfg.get("max_entries", 5)),
            )
        }

    memory_cfg = resolve_context_block(context_cfg.get("memory"))
    if memory_cfg.get("enabled", False):
        payload["memory"] = {
            "recent_decisions": _recent_decision_records(
                symbol=current_symbol,
                decision_history=decision_history,
                max_entries=int(memory_cfg.get("max_entries", 5)),
            )
        }

    news_cfg = resolve_context_block(context_cfg.get("news"))
    if news_cfg.get("enabled", False):
        payload["news"] = {
            "headlines": _recent_news_items(
                history=history,
                max_items=int(news_cfg.get("max_items", 5)),
            )
        }

    notes_cfg = resolve_context_block(context_cfg.get("notes"))
    if notes_cfg.get("enabled", False) and notes_payload is not None:
        payload["notes"] = dict(notes_payload)

    return payload
