"""Shared helpers for streaming bootstrap and deterministic replay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import re
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReplayWindow:
    start_date: str
    end_date: str
    pace_delay_ms: int = 0


def parse_iso_date(value: str, *, field_name: str) -> date:
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be in ISO format YYYY-MM-DD. Received: {value!r}."
        ) from exc


def parse_timeframe(timeframe: str) -> tuple[int, str] | None:
    normalized = str(timeframe or "").strip().lower()
    match = re.match(r"^(\d+)(mo|wk|m|h|d)$", normalized)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def timeframe_to_pandas_freq(timeframe: str) -> str:
    parsed = parse_timeframe(timeframe)
    if parsed is None:
        return "1D"
    amount, unit = parsed
    unit_map = {
        "m": "min",
        "h": "h",
        "d": "D",
        "wk": "W",
        "mo": "ME",
    }
    return f"{amount}{unit_map[unit]}"


def bootstrap_window_delta(timeframe: str, bars: int) -> timedelta:
    parsed = parse_timeframe(timeframe)
    multiplier = max(1, bars) * 3
    if parsed is None:
        return timedelta(days=multiplier)

    amount, unit = parsed
    steps = amount * multiplier
    if unit == "m":
        return timedelta(minutes=steps)
    if unit == "h":
        return timedelta(hours=steps)
    if unit == "d":
        return timedelta(days=steps)
    if unit == "wk":
        return timedelta(weeks=steps)
    return timedelta(days=steps * 30)


def bucket_for_timestamp(timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    freq = timeframe_to_pandas_freq(timeframe)
    if freq.endswith("W") or freq.endswith("ME"):
        bucket = timestamp.tz_convert(None) if timestamp.tzinfo else timestamp
        bucket = bucket.to_period(freq).to_timestamp()
        if timestamp.tzinfo:
            return bucket.tz_localize(timestamp.tzinfo)
        return bucket
    return timestamp.floor(freq)


def ensure_utc_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df if isinstance(df.index, pd.DatetimeIndex) else df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize(timezone.utc)
    else:
        out.index = out.index.tz_convert(timezone.utc)
    return out.sort_index()


def build_streaming_runtime_model_config(
    model_cfg: dict[str, Any],
    *,
    bootstrap_bars: int,
    end_date: str | None = None,
    replay_start_date: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    runtime_model_cfg = deepcopy(model_cfg)
    data_cfg = dict(runtime_model_cfg.get("data") or {})
    clock = now or datetime.now(timezone.utc)
    timeframe = str(data_cfg.get("timeframe") or "1d")

    effective_end = end_date or clock.date().isoformat()
    anchor_start = replay_start_date or effective_end
    recent_start = parse_iso_date(
        anchor_start,
        field_name=(
            "data.streaming.replay.start_date" if replay_start_date else "data.end_date"
        ),
    ) - bootstrap_window_delta(timeframe, bootstrap_bars)

    start_raw = data_cfg.get("start_date")
    try:
        configured_start = parse_iso_date(str(start_raw), field_name="data.start_date")
    except (TypeError, ValueError):
        configured_start = recent_start

    data_cfg["start_date"] = min(configured_start, recent_start).isoformat()
    data_cfg["end_date"] = effective_end
    data_cfg["test_start"] = None
    data_cfg["test_end"] = None
    runtime_model_cfg["data"] = data_cfg
    return runtime_model_cfg


def seed_history_frames(
    frames: dict[str, pd.DataFrame],
    *,
    history_window: int,
) -> dict[str, pd.DataFrame]:
    seeded: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        history = ensure_utc_datetime_index(frame)
        history = history[
            [
                column
                for column in ("Open", "High", "Low", "Close", "Volume")
                if column in history.columns
            ]
        ].copy()
        if history.empty:
            continue
        seeded[symbol] = history.tail(history_window)
    return seeded


def split_replay_frames(
    frames: dict[str, pd.DataFrame],
    *,
    replay_window: ReplayWindow,
    history_window: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    replay_start = pd.Timestamp(replay_window.start_date).tz_localize(timezone.utc)
    replay_end_exclusive = pd.Timestamp(replay_window.end_date).tz_localize(
        timezone.utc
    ) + timedelta(days=1)
    bootstrap_frames: dict[str, pd.DataFrame] = {}
    replay_frames: dict[str, pd.DataFrame] = {}
    symbols_manifest: list[dict[str, Any]] = []

    for symbol, frame in frames.items():
        history = ensure_utc_datetime_index(frame)
        replay_slice = history[
            (history.index >= replay_start) & (history.index < replay_end_exclusive)
        ]
        bootstrap_slice = history[history.index < replay_start].tail(history_window)
        if not bootstrap_slice.empty:
            bootstrap_frames[symbol] = bootstrap_slice
        if not replay_slice.empty:
            replay_frames[symbol] = replay_slice
        symbols_manifest.append(
            {
                "symbol": symbol,
                "bootstrap_bars": int(len(bootstrap_slice)),
                "replay_bars": int(len(replay_slice)),
                "replay_start": (
                    replay_slice.index.min().isoformat()
                    if not replay_slice.empty
                    else None
                ),
                "replay_end": (
                    replay_slice.index.max().isoformat()
                    if not replay_slice.empty
                    else None
                ),
            }
        )

    manifest = {
        "source": "replay",
        "start_date": replay_window.start_date,
        "end_date": replay_window.end_date,
        "pace_delay_ms": replay_window.pace_delay_ms,
        "symbols": symbols_manifest,
    }
    return bootstrap_frames, replay_frames, manifest
