from __future__ import annotations

"""Automatic reconnection utilities for streaming."""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RecoveryManager:
    """Handle reconnection attempts with exponential backoff."""

    max_attempts: int = 5
    base_delay: float = 1.0

    async def reconnect(
        self, name: str, connect: Optional[Callable[[], Awaitable[None]]] = None
    ) -> bool:
        """Attempt to reconnect using exponential backoff.

        Parameters
        ----------
        name:
            Identifier of the connection being recovered.
        connect:
            Awaitable returning a new connection when successful.  If omitted the
            method simply waits according to the backoff schedule and assumes
            success, making it suitable for tests.
        """

        delay = self.base_delay
        for attempt in range(self.max_attempts):
            try:
                if connect is not None:
                    await connect()
                logger.info("reconnect_success", name=name, attempt=attempt)
                return True
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.warning(
                    "reconnect_failed", name=name, attempt=attempt, error=str(exc)
                )
            jitter = random.random()
            await asyncio.sleep(delay + jitter)
            delay *= 2
        logger.error("reconnect_exhausted", name=name)
        return False
