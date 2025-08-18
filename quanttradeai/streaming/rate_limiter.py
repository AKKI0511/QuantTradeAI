"""Adaptive rate limiting utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from asyncio_throttle import Throttler


@dataclass
class AdaptiveRateLimiter:
    """Token bucket based rate limiter with burst handling."""

    base_rate: int
    burst_size: int

    def __post_init__(self) -> None:
        self.throttler = Throttler(rate_limit=self.base_rate, period=1)
        self._burst = asyncio.Semaphore(self.burst_size)

    async def acquire(self) -> None:
        async with self._burst:
            await self.throttler.acquire()
