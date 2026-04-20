"""Broker abstractions used by real-time agent execution backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _serialize_timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass(frozen=True)
class BrokerAccountSnapshot:
    account_id: str
    cash: float
    equity: float
    buying_power: float
    currency: str = "USD"
    status: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "cash": self.cash,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "currency": self.currency,
            "status": self.status,
        }


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    symbol: str
    qty: int
    market_price: float
    avg_entry_price: float
    side: str = "long"
    unrealized_pnl: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "market_price": self.market_price,
            "avg_entry_price": self.avg_entry_price,
            "side": self.side,
            "unrealized_pnl": self.unrealized_pnl,
        }


@dataclass(frozen=True)
class BrokerOrderResult:
    order_id: str
    symbol: str
    action: str
    qty: int
    status: str
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    filled_qty: int = 0
    filled_avg_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "action": self.action,
            "qty": self.qty,
            "status": self.status,
            "submitted_at": _serialize_timestamp(self.submitted_at),
            "filled_at": _serialize_timestamp(self.filled_at),
            "filled_qty": self.filled_qty,
            "filled_avg_price": self.filled_avg_price,
        }


class BrokerError(RuntimeError):
    """Raised when a broker request or state sync fails."""


class BrokerCredentialsError(BrokerError):
    """Raised when required broker credentials are missing."""


class BrokerClient(ABC):
    """Provider-agnostic broker interface."""

    provider: str = "unknown"

    @abstractmethod
    def get_account(self) -> BrokerAccountSnapshot:
        """Return the latest broker account snapshot."""

    @abstractmethod
    def list_positions(self) -> list[BrokerPositionSnapshot]:
        """Return the latest open positions."""

    @abstractmethod
    def submit_market_order(
        self,
        *,
        symbol: str,
        action: str,
        qty: int,
    ) -> BrokerOrderResult:
        """Submit a market order and return the broker's immediate response."""

    @abstractmethod
    def get_order(self, order_id: str) -> BrokerOrderResult:
        """Fetch a broker order by id."""

    @abstractmethod
    def wait_for_order(
        self,
        order_id: str,
        *,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> BrokerOrderResult:
        """Poll until the order reaches a terminal state or times out."""
