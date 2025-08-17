"""Manage multiple WebSocket connections."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List

from .adapters.base_adapter import DataProviderAdapter

Callback = Callable[[str, dict], Awaitable[None]]


@dataclass
class WebSocketManager:
    """Handle connection life-cycle for multiple adapters."""

    reconnect_attempts: int = 5
    adapters: List[DataProviderAdapter] = field(default_factory=list)

    def add_adapter(self, adapter: DataProviderAdapter) -> None:
        """Register a new data provider adapter."""

        self.adapters.append(adapter)

    async def _connect_with_retry(self, adapter: DataProviderAdapter) -> None:
        delay = 1.0
        for attempt in range(self.reconnect_attempts):
            try:
                await adapter.connect()
                return
            except Exception:
                if attempt == self.reconnect_attempts - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

    async def connect_all(self) -> None:
        """Connect all registered adapters."""

        for adapter in self.adapters:
            await self._connect_with_retry(adapter)

    async def run(self, callback: Callback) -> None:
        """Start listening on all connections and dispatch to ``callback``."""

        async def listen(adapter: DataProviderAdapter) -> None:
            async for msg in adapter.listen():
                await callback(adapter.name, msg)

        await asyncio.gather(*(listen(ad) for ad in self.adapters))
