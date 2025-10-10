"""Foundational components for pluggable streaming providers."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Set

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Base exception for provider related errors."""


class ProviderDiscoveryError(ProviderError):
    """Raised when provider discovery fails."""


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration validation fails."""


class ProviderConnectionError(ProviderError):
    """Raised when provider connections encounter unrecoverable errors."""


@dataclass(frozen=True)
class ProviderCapabilities:
    """Structured description of a provider's supported features."""

    asset_types: Set[str] = field(default_factory=set)
    data_types: Set[str] = field(default_factory=set)
    max_subscriptions: Optional[int] = None
    rate_limit_per_minute: Optional[int] = None
    requires_authentication: bool = False
    supports_order_book: bool = False
    supports_historical_data: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketDataEvent:
    """Base representation for normalized market data events."""

    provider: str
    symbol: str
    event_type: str
    timestamp: datetime
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - simple normalization
        if self.timestamp.tzinfo is None:
            object.__setattr__(
                self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc)
            )


@dataclass(frozen=True)
class QuoteEvent(MarketDataEvent):
    """Normalized quote event."""

    bid: Decimal = Decimal("0")
    ask: Decimal = Decimal("0")
    bid_size: float = 0.0
    ask_size: float = 0.0


@dataclass(frozen=True)
class TradeEvent(MarketDataEvent):
    """Normalized trade event."""

    price: Decimal = Decimal("0")
    size: float = 0.0
    trade_id: Optional[str] = None


@dataclass(frozen=True)
class OrderBookEvent(MarketDataEvent):
    """Normalized order book snapshot or delta."""

    bids: Sequence[Sequence[Decimal]] = field(default_factory=tuple)
    asks: Sequence[Sequence[Decimal]] = field(default_factory=tuple)
    depth: Optional[int] = None


@dataclass
class ProviderHealthStatus:
    """Runtime health metrics emitted by providers."""

    provider: str
    status: str = "initializing"
    latency_ms: float = 0.0
    error_rate: float = 0.0
    last_error: Optional[str] = None
    reconnect_attempts: int = 0
    circuit_open: bool = False
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: Dict[str, Any] = field(default_factory=dict)


class StreamingProviderAdapter(ABC):
    """Abstract base class for streaming provider integrations."""

    provider_name: str = "base"
    provider_version: str = "0.0.0"
    provider_description: str = ""
    provider_dependencies: Sequence[str] = ()
    default_capabilities: ProviderCapabilities = ProviderCapabilities()

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None) -> None:
        self._config: Dict[str, Any] = dict(config or {})
        self._subscriptions: Set[str] = set()
        self._connected: bool = False
        self._lock = asyncio.Lock()
        self._health_status = ProviderHealthStatus(provider=self.provider_name)

    # ------------------------------------------------------------------
    @property
    def config(self) -> Mapping[str, Any]:
        """Return the active configuration for the adapter."""

        return self._config

    @property
    def subscriptions(self) -> Set[str]:
        """Return currently subscribed symbols."""

        return set(self._subscriptions)

    @property
    def is_connected(self) -> bool:
        """Whether the adapter is connected to the upstream provider."""

        return self._connected

    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Establish the connection to the provider."""

        raise NotImplementedError

    async def disconnect(self) -> None:
        """Disconnect from the provider."""

        raise NotImplementedError

    async def subscribe(self, symbols: Sequence[str]) -> None:
        """Subscribe to the requested market data streams."""

        raise NotImplementedError

    async def unsubscribe(self, symbols: Sequence[str]) -> None:
        """Unsubscribe from the provided market data streams."""

        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return the provider's capabilities."""

    def validate_config(self, config: Mapping[str, Any]) -> MutableMapping[str, Any]:
        """Validate the adapter configuration and return a normalized mapping."""

        return dict(config)

    def get_health_status(self) -> ProviderHealthStatus:
        """Return the most recently recorded health status."""

        return self._health_status

    # ------------------------------------------------------------------
    async def _mark_connected(self) -> None:
        async with self._lock:
            self._connected = True
            self._health_status.status = "connected"
            self._health_status.last_check = datetime.now(timezone.utc)

    async def _mark_disconnected(self, *, reason: Optional[str] = None) -> None:
        async with self._lock:
            self._connected = False
            self._health_status.status = "disconnected"
            self._health_status.last_error = reason
            self._health_status.last_check = datetime.now(timezone.utc)

    async def _record_latency(self, latency_ms: float) -> None:
        async with self._lock:
            self._health_status.latency_ms = latency_ms
            self._health_status.last_check = datetime.now(timezone.utc)

    async def _record_error(self, error: Exception) -> None:
        async with self._lock:
            self._health_status.last_error = str(error)
            self._health_status.status = "error"
            self._health_status.reconnect_attempts += 1
            self._health_status.last_check = datetime.now(timezone.utc)
            logger.exception(
                "provider_operation_failed", extra={"provider": self.provider_name}
            )

    # ------------------------------------------------------------------
    @classmethod
    def metadata(cls) -> Dict[str, Any]:
        """Return static metadata for discovery without instantiation."""

        return {
            "name": cls.provider_name,
            "version": cls.provider_version,
            "description": cls.provider_description,
            "dependencies": tuple(cls.provider_dependencies),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}(provider='{self.provider_name}', version='{self.provider_version}')"
