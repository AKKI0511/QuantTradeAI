"""Portfolio management helpers.

Provides the :class:`PortfolioManager` class which tracks cash, positions
and risk exposure for multiple symbols.

Key Components:
    - :class:`PortfolioManager`

Typical Usage:
    ```python
    from quanttradeai.trading import PortfolioManager
    pm = PortfolioManager(capital=100000)
    qty = pm.open_position("AAPL", price=150, stop_loss_pct=0.02)
    ```
"""

from __future__ import annotations

from typing import Dict

from .risk import position_size


class PortfolioManager:
    """Manage capital allocation and risk across multiple symbols."""

    def __init__(
        self,
        capital: float,
        max_risk_per_trade: float = 0.02,
        max_portfolio_risk: float = 0.1,
    ) -> None:
        self.initial_capital = capital
        self.cash = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.positions: Dict[str, dict] = {}

    @property
    def portfolio_value(self) -> float:
        return self.cash + sum(p["qty"] * p["price"] for p in self.positions.values())

    @property
    def risk_exposure(self) -> float:
        if self.portfolio_value == 0:
            return 0.0
        exposure = sum(
            p["qty"] * p["price"] * p["stop_loss_pct"] for p in self.positions.values()
        )
        return exposure / self.portfolio_value

    def _remaining_risk_capacity(self) -> float:
        return self.max_portfolio_risk * self.portfolio_value - sum(
            p["qty"] * p["price"] * p["stop_loss_pct"] for p in self.positions.values()
        )

    def open_position(
        self, symbol: str, price: float, stop_loss_pct: float | None = None
    ) -> int:
        """Open a new position and return the quantity allocated."""
        if symbol in self.positions:
            raise ValueError(f"Position for {symbol} already exists")

        if stop_loss_pct is None or stop_loss_pct <= 0:
            qty = int(self.max_risk_per_trade * self.portfolio_value / price)
            stop_loss = 0.0
        else:
            qty = position_size(
                capital=self.portfolio_value,
                risk_per_trade=self.max_risk_per_trade,
                stop_loss_pct=stop_loss_pct,
                price=price,
            )
            allowed_qty = int(self._remaining_risk_capacity() / (price * stop_loss_pct))
            qty = min(qty, allowed_qty)
            stop_loss = stop_loss_pct

        qty = min(qty, int(self.cash // price))
        if qty <= 0:
            return 0

        self.cash -= qty * price
        self.positions[symbol] = {
            "qty": qty,
            "price": price,
            "stop_loss_pct": stop_loss,
        }
        return qty

    def close_position(self, symbol: str, price: float) -> int:
        """Close an existing position and return quantity closed."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return 0
        self.cash += pos["qty"] * price
        return pos["qty"]
