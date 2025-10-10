"""Example streaming provider adapter used for testing and documentation."""

from __future__ import annotations

import asyncio
from typing import Mapping, MutableMapping, Sequence

from ..providers.base import (
    ProviderCapabilities,
    ProviderHealthStatus,
    StreamingProviderAdapter,
)


class ExampleStreamingProvider(StreamingProviderAdapter):
    """In-memory streaming provider used for unit testing."""

    provider_name = "example"
    provider_version = "1.0.0"
    provider_description = "Reference adapter demonstrating the provider interface"

    def __init__(self, *, config: Mapping[str, object] | None = None) -> None:
        super().__init__(config=config)
        self._connected_event = asyncio.Event()

    async def connect(self) -> None:
        await asyncio.sleep(0)
        await self._mark_connected()
        self._connected_event.set()

    async def disconnect(self) -> None:
        await asyncio.sleep(0)
        await self._mark_disconnected()
        self._connected_event.clear()

    async def subscribe(self, symbols: Sequence[str]) -> None:
        await self._connected_event.wait()
        self._subscriptions.update(symbols)

    async def unsubscribe(self, symbols: Sequence[str]) -> None:
        await self._connected_event.wait()
        for symbol in symbols:
            self._subscriptions.discard(symbol)

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            asset_types={"stocks", "crypto"},
            data_types={"trades", "quotes", "order_book"},
            max_subscriptions=500,
            rate_limit_per_minute=1200,
            requires_authentication=False,
            supports_order_book=True,
            metadata={"example": True},
        )

    def validate_config(
        self, config: Mapping[str, object]
    ) -> MutableMapping[str, object]:
        normalized = dict(config)
        normalized.setdefault("options", {})
        normalized.setdefault("credentials", {})
        return normalized

    def get_health_status(self) -> ProviderHealthStatus:
        status = super().get_health_status()
        status.metrics["subscriptions"] = len(self._subscriptions)
        return status
