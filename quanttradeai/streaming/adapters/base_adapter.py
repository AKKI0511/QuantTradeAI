"""Base classes for data provider adapters."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import websockets


@dataclass
class DataProviderAdapter(ABC):
    """Abstract adapter defining basic WebSocket operations."""

    websocket_url: str
    name: str
    connection: Optional[websockets.WebSocketClientProtocol] = field(
        default=None, init=False
    )

    async def connect(self) -> None:
        """Establish a WebSocket connection to the provider."""

        self.connection = await websockets.connect(self.websocket_url)

    async def subscribe(self, channel: str, symbols: List[str]) -> None:
        """Send a subscription message for ``symbols`` on ``channel``."""

        if self.connection is None:
            await self.connect()
        message = self._build_subscribe_message(channel, symbols)
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
