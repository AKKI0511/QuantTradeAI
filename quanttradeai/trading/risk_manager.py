"""Risk management coordinator."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .drawdown_guard import DrawdownGuard


class RiskManager:
    """Coordinate risk guards like :class:`DrawdownGuard`."""

    def __init__(self, drawdown_guard: Optional[DrawdownGuard] = None) -> None:
        self.drawdown_guard = drawdown_guard

    def update(self, portfolio_value: float, timestamp: datetime) -> None:
        if self.drawdown_guard is not None:
            self.drawdown_guard.update_portfolio_value(portfolio_value, timestamp)

    def record_trade(self, notional: float, timestamp: datetime) -> None:
        if self.drawdown_guard is not None:
            self.drawdown_guard.record_trade(notional, timestamp)

    def get_position_size_multiplier(self) -> float:
        if self.drawdown_guard is not None:
            return self.drawdown_guard.get_position_size_multiplier()
        return 1.0

    def should_halt_trading(self) -> bool:
        if self.drawdown_guard is not None:
            return self.drawdown_guard.should_halt_trading()
        return False

    def should_emergency_liquidate(self) -> bool:
        if self.drawdown_guard is not None:
            return self.drawdown_guard.should_emergency_liquidate()
        return False

    def get_risk_metrics(self) -> dict:
        if self.drawdown_guard is not None:
            return self.drawdown_guard.get_risk_metrics()
        return {}
