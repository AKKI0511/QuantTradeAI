"""Base classes for data provider adapters."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import websockets

from ..auth_manager import AuthManager
from ..rate_limiter import AdaptiveRateLimiter


@dataclass
class DataProviderAdapter(ABC):
    """Abstract adapter defining basic WebSocket operations."""

    websocket_url: str
    name: str
    connection: Optional[websockets.WebSocketClientProtocol] = field(
        default=None, init=False
    )
    circuit_breaker: Any = field(default=None, init=False)
    rate_limiter: Optional[AdaptiveRateLimiter] = field(default=None, init=False)
    auth_manager: Optional[AuthManager] = field(default=None, init=False)

    async def connect(self) -> None:
        """Establish a WebSocket connection to the provider."""

        headers = None
        if self.auth_manager:
            headers = await self.auth_manager.get_auth_headers()
        self.connection = await websockets.connect(
            self.websocket_url, extra_headers=headers
        )

    async def subscribe(self, channel: str, symbols: List[str]) -> None:
        """Send a subscription message for ``symbols`` on ``channel``."""

        if self.connection is None:
            await self.connect()
        message = self._build_subscribe_message(channel, symbols)
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        await self.connection.send(json.dumps(message))

    async def listen(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield parsed JSON messages from the connection."""

        if self.connection is None:
            raise RuntimeError("Connection not established")
        async for message in self.connection:
            yield json.loads(message)

    @abstractmethod
    def _build_subscribe_message(
        self, channel: str, symbols: List[str]
    ) -> Dict[str, Any]:
        """Return provider specific subscription payload."""
