"""Deterministic historical replay gateway for paper trading."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .stream_buffer import StreamBuffer
from .history import ensure_utc_datetime_index


@dataclass
class ReplayGateway:
    """Replay historical OHLCV bars into the standard stream buffer."""

    frames: dict[str, pd.DataFrame]
    pace_delay_ms: int = 0
    buffer_size: int = 1000
    buffer: StreamBuffer = field(init=False)
    flush_on_stop: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self.buffer = StreamBuffer(self.buffer_size)

    def _iter_messages(self) -> list[tuple[pd.Timestamp, int, dict[str, Any]]]:
        events: list[tuple[pd.Timestamp, int, dict[str, Any]]] = []
        order = 0
        for symbol, frame in sorted(self.frames.items()):
            history = ensure_utc_datetime_index(frame)
            for timestamp, row in history.iterrows():
                close = float(row.get("Close", row.get("close", 0.0)))
                events.append(
                    (
                        timestamp,
                        order,
                        {
                            "type": "replay_bar",
                            "symbol": symbol,
                            "timestamp": timestamp.isoformat(),
                            "open": float(row.get("Open", row.get("open", close))),
                            "high": float(row.get("High", row.get("high", close))),
                            "low": float(row.get("Low", row.get("low", close))),
                            "close": close,
                            "price": close,
                            "volume": float(row.get("Volume", row.get("volume", 0.0))),
                        },
                    )
                )
                order += 1
        events.sort(key=lambda item: (item[0], item[1]))
        return events

    async def _start(self) -> None:
        delay_seconds = max(self.pace_delay_ms, 0) / 1000.0
        for _timestamp, _order, message in self._iter_messages():
            await self.buffer.put(message)
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
