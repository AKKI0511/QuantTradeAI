"""Health monitoring utilities."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class HealthMonitor:
    """Periodically perform a no-op health check."""

    interval: int = 30

    async def run(self) -> None:
        """Run the health check loop."""

        while True:
            await asyncio.sleep(self.interval)
