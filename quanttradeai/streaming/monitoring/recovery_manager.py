from __future__ import annotations

"""Automatic reconnection utilities for streaming."""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional
from pybreaker import CircuitBreaker, CircuitBreakerError

from quanttradeai.streaming.logging import logger


@dataclass
class RecoveryManager:
    """Handle reconnection attempts with exponential backoff and circuit breaking."""

    max_attempts: int = 5
    base_delay: float = 1.0
    reset_timeout: float = 60.0
    circuit_breaker: CircuitBreaker = field(init=False)

    def __post_init__(self) -> None:
        self.circuit_breaker = CircuitBreaker(
            fail_max=self.max_attempts, reset_timeout=self.reset_timeout
        )

    async def reconnect(
        self, name: str, connect: Optional[Callable[[], Awaitable[None]]] = None
    ) -> bool:
        """Attempt to reconnect using exponential backoff.

        If reconnection keeps failing ``max_attempts`` times the internal circuit
        breaker opens and further attempts immediately fail until the
        ``reset_timeout`` elapses.
        """

        delay = self.base_delay
        if connect is None:

            async def connect():
                return None

        for attempt in range(self.max_attempts):
            try:
                await self.circuit_breaker.call_async(connect)
                logger.info("reconnect_success", name=name, attempt=attempt)
                return True
            except CircuitBreakerError:
                logger.error("circuit_open", name=name)
                return False
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.warning(
                    "reconnect_failed", name=name, attempt=attempt, error=str(exc)
                )
            jitter = random.random()
            await asyncio.sleep(delay + jitter)
            delay *= 2
        logger.error("reconnect_exhausted", name=name)
        return False
