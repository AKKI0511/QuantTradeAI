"""Backtest engine coordinating portfolio and risk management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.trading.risk_manager import RiskManager

from .backtester import simulate_trades


@dataclass
class BacktestEngine:
    """Simple wrapper around :func:`simulate_trades`.

    Parameters
    ----------
    portfolio: PortfolioManager | None
        Portfolio manager used for capital allocation when backtesting multiple
        symbols.
    risk_manager: RiskManager | None
        Risk manager coordinating guards during backtests.
    """

    portfolio: Optional[PortfolioManager] = None
    risk_manager: Optional[RiskManager] = None

    def run(
        self,
        data: pd.DataFrame | Dict[str, pd.DataFrame],
        execution: dict | None = None,
        **kwargs,
    ) -> pd.DataFrame | Dict[str, pd.DataFrame]:
        """Execute a backtest using the underlying :func:`simulate_trades`.

        Any additional keyword arguments are forwarded to
        :func:`simulate_trades`.
        """
        if self.portfolio is not None and self.risk_manager is not None:
            # ensure portfolio uses provided risk manager
            self.portfolio.risk_manager = self.risk_manager
        return simulate_trades(
            data,
            execution=execution,
            portfolio=self.portfolio,
            drawdown_guard=(
                self.risk_manager.drawdown_guard if self.risk_manager else None
            ),
            **kwargs,
        )
