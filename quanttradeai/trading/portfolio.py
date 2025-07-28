# Portfolio management module
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .risk import position_size


@dataclass
class Portfolio:
    """Simple portfolio manager for backtests."""

    capital: float
    risk_per_trade: float
    cash: float | None = None
    positions: Dict[str, int] = field(default_factory=dict)
    _value: float | None = None

    def __post_init__(self) -> None:
        self.cash = self.capital if self.cash is None else self.cash
        self._value = self.capital if self._value is None else self._value

    @property
    def total_value(self) -> float:
        return self._value if self._value is not None else 0.0

    def allocate(
        self,
        symbol: str,
        price: float,
        stop_loss_pct: float,
        direction: int = 1,
        trade_cost: float = 0.0,
    ) -> int:
        """Open a new position and return allocated quantity."""
        qty = position_size(self.total_value, self.risk_per_trade, stop_loss_pct, price)
        if direction > 0 and qty * price > self.cash:
            qty = int(self.cash // price)
        qty *= 1 if direction >= 0 else -1
        self.cash -= qty * price
        self.cash -= trade_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + qty
        return qty

    def close(self, symbol: str, price: float, trade_cost: float = 0.0) -> int:
        """Close an existing position and return closed quantity."""
        qty = self.positions.pop(symbol, 0)
        self.cash += qty * price
        self.cash -= trade_cost
        return qty

    def update_value(self, prices: Dict[str, float]) -> None:
        holdings = sum(
            qty * prices.get(sym, 0.0) for sym, qty in self.positions.items()
        )
        self._value = self.cash + holdings
