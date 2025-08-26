"""Connection pooling and health management."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Set, Callable, Awaitable


@dataclass
class ConnectionPool:
    """Maintain a pool of reusable WebSocket connections."""

    max_connections: int = 10
    _pool: asyncio.Queue = field(init=False)
    _active: Set[Any] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self._pool = asyncio.Queue(maxsize=self.max_connections)

    async def acquire_connection(self, factory: Callable[[], Awaitable[Any]]) -> Any:
        if not self._pool.empty():
            conn = await self._pool.get()
        elif len(self._active) < self.max_connections:
            conn = await factory()
        else:
            conn = await self._pool.get()
        self._active.add(conn)
        return conn

    async def release(self, conn: Any) -> None:
        if conn in self._active:
            self._active.remove(conn)
            await self._pool.put(conn)

    async def _ping_all_connections(self) -> None:
        for conn in list(self._active):
            if hasattr(conn, "ping"):
                await conn.ping()

    async def health_check_loop(self, interval: int = 30) -> None:
        while True:
            await self._ping_all_connections()
            await asyncio.sleep(interval)
