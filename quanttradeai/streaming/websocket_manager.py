"""Manage multiple WebSocket connections."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from pybreaker import CircuitBreaker, CircuitBreakerError

from .adapters.base_adapter import DataProviderAdapter
from .auth_manager import AuthManager
from .rate_limiter import AdaptiveRateLimiter
from .connection_pool import ConnectionPool
from .logging import logger

Callback = Callable[[str, dict], Awaitable[None]]


@dataclass
class WebSocketManager:
    """Handle connection life-cycle for multiple adapters."""

    reconnect_attempts: int = 5
    adapters: List[DataProviderAdapter] = field(default_factory=list)
    connection_pool: ConnectionPool = field(default_factory=ConnectionPool)

    def add_adapter(
        self,
        adapter: DataProviderAdapter,
        *,
        circuit_breaker_cfg: Optional[dict] = None,
        rate_limit_cfg: Optional[dict] = None,
        auth_method: str = "api_key",
    ) -> None:
        """Register a new data provider adapter with optional safeguards."""

        cb_cfg = circuit_breaker_cfg or {}
        adapter.circuit_breaker = CircuitBreaker(
            fail_max=cb_cfg.get("failure_threshold", 5),
            reset_timeout=cb_cfg.get("timeout", 30),
        )
        if rate_limit_cfg:
            adapter.rate_limiter = AdaptiveRateLimiter(
                base_rate=rate_limit_cfg.get("default_rate", 100),
                burst_size=rate_limit_cfg.get("burst_allowance", 50),
            )
        if auth_method != "none":
            adapter.auth_manager = AuthManager(adapter.name)
        self.adapters.append(adapter)

    async def _connect_with_retry(self, adapter: DataProviderAdapter) -> None:
        delay = 1.0
        for attempt in range(self.reconnect_attempts):
            try:
                await adapter.circuit_breaker.call_async(adapter.connect)
                return
            except CircuitBreakerError as exc:
                logger.error("circuit_open", provider=adapter.name, error=str(exc))
                raise
            except Exception as exc:
                logger.warning(
                    "connect_failed",
                    provider=adapter.name,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt == self.reconnect_attempts - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2

    async def connect_all(self) -> None:
        """Connect all registered adapters."""

        for adapter in self.adapters:
            await self._connect_with_retry(adapter)
        self.connection_pool._active.update(
            ad.connection for ad in self.adapters if ad.connection
        )
        self.connection_pool._pool = asyncio.Queue()
        for ad in self.adapters:
            if ad.connection:
                await self.connection_pool._pool.put(ad.connection)
        logger.info("connections_established", count=len(self.adapters))

    async def run(self, callback: Callback) -> None:
        """Start listening on all connections and dispatch to ``callback``."""

        async def listen(adapter: DataProviderAdapter) -> None:
            async for msg in adapter.listen():
                await callback(adapter.name, msg)

        await asyncio.gather(*(listen(ad) for ad in self.adapters))
