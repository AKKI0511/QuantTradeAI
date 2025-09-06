"""Real-time position management with intraday risk controls.

Bridges the streaming gateway with :class:`RiskManager` and market impact
models to maintain up-to-date positions, evaluate intraday risk and provide
execution quality analytics. The implementation follows the project's
dataclass-based architecture and uses thread-safe operations for
concurrent streaming callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import threading
from typing import Any, Dict, List

import yaml

try:
    from quanttradeai.streaming.gateway import StreamingGateway
    from quanttradeai.streaming.logging import logger
except Exception:  # pragma: no cover - optional streaming deps
    StreamingGateway = Any  # type: ignore
    import logging

    logger = logging.getLogger(__name__)
from quanttradeai.trading.drawdown_guard import DrawdownGuard
from quanttradeai.trading.risk_manager import RiskManager
from quanttradeai.backtest.impact import ImpactCalculator, MODEL_MAP
from quanttradeai.utils.config_schemas import (
    PositionManagerConfig,
    RiskManagementConfig,
    MarketImpactConfig,
)


@dataclass
class Position:
    """Represents an open position for a symbol."""

    qty: int = 0
    avg_price: float = 0.0
    market_price: float = 0.0

    def value(self) -> float:
        return self.qty * self.market_price

    def update(self, qty: int, price: float) -> None:
        new_qty = self.qty + qty
        if new_qty == 0:
            self.qty = 0
            self.avg_price = 0.0
            return
        self.avg_price = (self.avg_price * self.qty + price * qty) / new_qty
        self.qty = new_qty


@dataclass
class ExecutionRecord:
    """Execution bookkeeping used for analytics."""

    symbol: str
    qty: int
    price: float
    timestamp: datetime
    impact_cost: float = 0.0


@dataclass
class PositionManager:
    """Track positions and enforce intraday risk controls.

    Parameters
    ----------
    risk_manager:
        Optional :class:`RiskManager` coordinating drawdown and turnover
        limits.
    impact:
        Optional :class:`ImpactCalculator` for execution cost estimation.
    reconciliation:
        Mapping of timeframe labels (e.g. ``{"intraday": "1m"}``) used by
        :meth:`reconcile_positions`.
    mode:
        ``"paper"`` or ``"live"`` to indicate execution environment.
    cash:
        Starting cash balance for portfolio value calculations.
    """

    risk_manager: RiskManager | None = None
    impact: ImpactCalculator | None = None
    reconciliation: Dict[str, str] = field(
        default_factory=lambda: {"intraday": "1m", "daily": "1d"}
    )
    mode: str = "paper"
    cash: float = 0.0

    _positions: Dict[str, Position] = field(default_factory=dict, init=False)
    _executions: List[ExecutionRecord] = field(default_factory=list, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    # ------------------------------------------------------------------
    # Construction utilities
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls, config: PositionManagerConfig | Dict | str | None = None
    ) -> "PositionManager":
        """Instantiate from a YAML file or config object."""

        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config = data.get("position_manager", data)
        if isinstance(config, dict) or config is None:
            config = PositionManagerConfig(**(config or {}))

        rm: RiskManager | None = None
        rm_cfg: RiskManagementConfig = config.risk_management
        if rm_cfg.drawdown_protection.enabled or rm_cfg.turnover_limits != {}:
            rm = RiskManager(DrawdownGuard(rm_cfg))

        impact_calc: ImpactCalculator | None = None
        imp_cfg: MarketImpactConfig = config.impact
        if imp_cfg.enabled:
            model_cls = MODEL_MAP.get(imp_cfg.model, MODEL_MAP["linear"])
            params: Dict[str, float] = {"alpha": imp_cfg.alpha, "beta": imp_cfg.beta}
            if imp_cfg.gamma is not None:
                params["gamma"] = imp_cfg.gamma
            model = model_cls(**params)
            impact_calc = ImpactCalculator(
                model=model, decay=imp_cfg.decay, spread=imp_cfg.spread
            )

        return cls(
            risk_manager=rm,
            impact=impact_calc,
            reconciliation=config.reconciliation,
            mode=config.mode,
        )

    # ------------------------------------------------------------------
    # Streaming integration
    # ------------------------------------------------------------------
    def bind_gateway(self, gateway: StreamingGateway, symbols: List[str]) -> None:
        """Subscribe to market data for ``symbols`` via ``gateway``."""

        gateway.subscribe_to_quotes(symbols, self.handle_market_data)
        gateway.subscribe_to_trades(symbols, self.handle_market_data)

    def handle_market_data(self, message: Dict[str, Any]) -> None:
        """Process incoming market data messages.

        Expects dictionary messages with at least ``symbol`` and ``price``
        fields. The risk manager is updated with the latest portfolio value.
        """

        symbol = message.get("symbol")
        price = message.get("price") or message.get("last") or message.get("close")
        ts = message.get("timestamp") or datetime.utcnow()
        if symbol is None or price is None:
            return

        with self._lock:
            pos = self._positions.get(symbol)
            if pos is not None:
                pos.market_price = price
            value = self.portfolio_value

        if self.risk_manager is not None:
            self.risk_manager.update(value, ts)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def open_position(
        self,
        symbol: str,
        qty: int,
        price: float,
        adv: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Open or increase a position."""

        if qty == 0:
            return
        ts = timestamp or datetime.utcnow()
        if self.risk_manager is not None:
            if self.risk_manager.should_halt_trading():
                logger.warning("Trade halted by risk guard for %s", symbol)
                return
            qty = int(qty * self.risk_manager.get_position_size_multiplier())
            if qty == 0:
                return
        with self._lock:
            pos = self._positions.setdefault(symbol, Position())
            pos.market_price = price
            pos.update(qty, price)
            notional = qty * price
            self.cash -= notional
            impact_cost = 0.0
            if self.impact is not None and adv is not None:
                impact_cost = self.impact.impact_cost(abs(qty), adv)["total"]
            self._executions.append(
                ExecutionRecord(
                    symbol=symbol,
                    qty=qty,
                    price=price,
                    timestamp=ts,
                    impact_cost=impact_cost,
                )
            )
            value = self.portfolio_value

        if self.risk_manager is not None:
            self.risk_manager.record_trade(notional, ts)
            self.risk_manager.update(value, ts)

    def close_position(
        self,
        symbol: str,
        price: float,
        adv: float | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        """Close a position and return quantity closed."""

        ts = timestamp or datetime.utcnow()
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None or pos.qty == 0:
                return 0
            if (
                self.risk_manager is not None
                and self.risk_manager.should_halt_trading()
            ):
                logger.warning("Trade halted by risk guard for %s", symbol)
                return 0
            qty = -pos.qty
            pos.update(qty, price)
            pos.market_price = price
            notional = -qty * price
            self.cash += notional
            impact_cost = 0.0
            if self.impact is not None and adv is not None:
                impact_cost = self.impact.impact_cost(abs(qty), adv)["total"]
            self._executions.append(
                ExecutionRecord(
                    symbol=symbol,
                    qty=qty,
                    price=price,
                    timestamp=ts,
                    impact_cost=impact_cost,
                )
            )
            if pos.qty == 0:
                del self._positions[symbol]
            value = self.portfolio_value

        if self.risk_manager is not None:
            self.risk_manager.record_trade(notional, ts)
            self.risk_manager.update(value, ts)
        return -qty

    # ------------------------------------------------------------------
    # Analytics & reconciliation
    # ------------------------------------------------------------------
    @property
    def portfolio_value(self) -> float:
        with self._lock:
            return self.cash + sum(p.value() for p in self._positions.values())

    def reconcile_positions(
        self, now: datetime | None = None
    ) -> Dict[str, Dict[str, int]]:
        """Return net positions for configured timeframes."""

        now = now or datetime.utcnow()
        with self._lock:
            intraday = {s: p.qty for s, p in self._positions.items()}
            daily: Dict[str, int] = {}
            for rec in self._executions:
                if rec.timestamp.date() == now.date():
                    daily[rec.symbol] = daily.get(rec.symbol, 0) + rec.qty
        return {"intraday": intraday, "daily": daily}

    def execution_metrics(self) -> Dict[str, float]:
        """Aggregate basic execution quality metrics."""

        with self._lock:
            total_impact = sum(e.impact_cost for e in self._executions)
            trades = len(self._executions)
        return {"trades": trades, "total_impact_cost": total_impact}
