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

from typing import Dict, TYPE_CHECKING
from datetime import datetime, UTC

from .risk import position_size
from .risk_manager import RiskManager

if TYPE_CHECKING:
    from .drawdown_guard import DrawdownGuard


class PortfolioManager:
    """Manage capital allocation and risk across multiple symbols."""

    def __init__(
        self,
        capital: float,
        max_risk_per_trade: float = 0.02,
        max_portfolio_risk: float = 0.1,
        risk_manager: RiskManager | None = None,
        drawdown_guard: "DrawdownGuard | None" = None,
    ) -> None:
        """Initialize a portfolio manager.

        Parameters
        ----------
        capital:
            Initial capital allocated to the portfolio.
        max_risk_per_trade:
            Maximum fraction of the portfolio value risked per trade.
        max_portfolio_risk:
            Maximum fraction of the portfolio value that can be exposed at once.
        risk_manager:
            Optional pre-configured :class:`RiskManager` instance.
        drawdown_guard:
            Optional :class:`DrawdownGuard` that will be wrapped in a
            :class:`RiskManager`. Cannot be provided together with
            ``risk_manager``.
        """
        if risk_manager is not None and drawdown_guard is not None:
            raise ValueError("Provide either risk_manager or drawdown_guard, not both.")

        if drawdown_guard is not None:
            risk_manager = RiskManager(drawdown_guard=drawdown_guard)

        self.initial_capital = capital
        self.cash = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.positions: Dict[str, dict] = {}
        self.risk_manager = risk_manager

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

        if self.risk_manager is not None:
            self.risk_manager.update(self.portfolio_value, datetime.now(UTC))
            if self.risk_manager.should_emergency_liquidate():
                self.close_all_positions()
                return 0
            if self.risk_manager.should_halt_trading():
                return 0

        multiplier = (
            self.risk_manager.get_position_size_multiplier()
            if self.risk_manager is not None
            else 1.0
        )

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

        qty = int(qty * multiplier)
        qty = min(qty, int(self.cash // price))
        if qty <= 0:
            return 0

        notional = qty * price
        self.cash -= notional
        self.positions[symbol] = {
            "qty": qty,
            "price": price,
            "stop_loss_pct": stop_loss,
        }
        if self.risk_manager is not None:
            self.risk_manager.record_trade(notional, datetime.now(UTC))
        return qty

    def close_position(self, symbol: str, price: float) -> int:
        """Close an existing position and return quantity closed."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return 0
        notional = pos["qty"] * price
        self.cash += notional
        if self.risk_manager is not None:
            self.risk_manager.record_trade(notional, datetime.now(UTC))
        return pos["qty"]

    def close_all_positions(
        self, prices: Dict[str, float] | None = None
    ) -> Dict[str, int]:
        """Close all positions using provided or last known prices."""
        closed: Dict[str, int] = {}
        prices = prices or {}
        symbols = list(self.positions.keys())
        for symbol in symbols:
            price = prices.get(symbol, self.positions[symbol]["price"])
            qty = self.close_position(symbol, price)
            closed[symbol] = qty
        return closed
