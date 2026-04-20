"""Shared broker-backed execution helpers for real-time agent engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.trading.position_manager import PositionManager

from .alpaca import AlpacaBrokerClient
from .base import (
    BrokerAccountSnapshot,
    BrokerClient,
    BrokerError,
    BrokerOrderResult,
    BrokerPositionSnapshot,
)


def resolve_execution_backend(agent_config: dict[str, Any]) -> str:
    execution_cfg = dict(agent_config.get("execution") or {})
    return str(execution_cfg.get("backend") or "simulated").strip().lower()


def create_broker_client_for_agent(
    agent_config: dict[str, Any],
    *,
    mode: str,
) -> BrokerClient | None:
    backend = resolve_execution_backend(agent_config)
    if backend == "simulated":
        return None
    if backend == "alpaca":
        return AlpacaBrokerClient(mode=mode)
    raise ValueError(f"Unsupported execution backend: {backend}")


def _execution_status_from_order(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "filled":
        return "executed"
    if normalized == "partially_filled":
        return "partial_fill"
    if normalized in {"canceled", "expired", "done_for_day"}:
        return "canceled"
    if normalized == "rejected":
        return "rejected"
    return normalized or "submitted"


@dataclass
class BrokerExecutionRuntime:
    """Keep local execution state in sync with broker truth."""

    broker_client: BrokerClient
    portfolio: PortfolioManager
    position_manager: PositionManager | None
    stop_loss_pct: float
    provider: str | None = None
    start_account: dict[str, Any] | None = field(default=None, init=False)
    end_account: dict[str, Any] | None = field(default=None, init=False)
    start_positions: list[dict[str, Any]] | None = field(default=None, init=False)
    end_positions: list[dict[str, Any]] | None = field(default=None, init=False)
    starting_equity: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = getattr(self.broker_client, "provider", "unknown")

    def _serialized_positions(
        self,
        positions: list[BrokerPositionSnapshot],
    ) -> list[dict[str, Any]]:
        return [position.to_dict() for position in positions]

    def _apply_snapshots(
        self,
        *,
        account: BrokerAccountSnapshot,
        positions: list[BrokerPositionSnapshot],
    ) -> None:
        portfolio_positions: dict[str, dict[str, Any]] = {}
        realtime_positions: dict[str, dict[str, Any]] = {}
        for position in positions:
            side = str(position.side or "long").strip().lower()
            if side not in {"long", ""} and position.qty > 0:
                raise BrokerError(
                    f"Unsupported broker position for {position.symbol}: QuantTradeAI Alpaca execution only supports long/flat portfolios."
                )
            if position.qty <= 0:
                continue
            portfolio_positions[position.symbol] = {
                "qty": position.qty,
                "price": position.market_price,
                "entry_price": position.avg_entry_price,
                "stop_loss_pct": self.stop_loss_pct,
            }
            realtime_positions[position.symbol] = {
                "qty": position.qty,
                "avg_price": position.avg_entry_price,
                "market_price": position.market_price,
            }

        if self.starting_equity is None:
            self.starting_equity = account.equity

        self.portfolio.replace_state(
            cash=account.cash,
            positions=portfolio_positions,
            initial_capital=self.starting_equity,
        )
        if self.position_manager is not None:
            self.position_manager.replace_state(
                cash=account.cash,
                positions=realtime_positions,
            )

    def sync_from_broker(
        self,
        *,
        mark: str | None = None,
    ) -> tuple[BrokerAccountSnapshot, list[BrokerPositionSnapshot]]:
        account = self.broker_client.get_account()
        positions = self.broker_client.list_positions()
        self._apply_snapshots(account=account, positions=positions)
        if mark == "start":
            self.start_account = account.to_dict()
            self.start_positions = self._serialized_positions(positions)
        elif mark == "end":
            self.end_account = account.to_dict()
            self.end_positions = self._serialized_positions(positions)
        return account, positions

    def start_session(self) -> None:
        self.sync_from_broker(mark="start")

    def finish_session(self) -> None:
        self.sync_from_broker(mark="end")

    def execute_action(
        self,
        *,
        symbol: str,
        action: str,
        price: float,
        timestamp: datetime,
        extra: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        normalized_action = str(action or "").strip().lower()
        current_position = dict(self.portfolio.positions.get(symbol) or {})
        current_qty = int(current_position.get("qty", 0))

        if normalized_action == "buy":
            if current_qty > 0:
                return "already_long", None
            qty = self.portfolio.estimate_open_position_qty(
                price,
                stop_loss_pct=self.stop_loss_pct,
            )
            if qty <= 0:
                return "blocked", None
        elif normalized_action == "sell":
            if current_qty <= 0:
                return "no_position", None
            qty = current_qty
        else:
            return "hold", None

        initial_order = self.broker_client.submit_market_order(
            symbol=symbol,
            action=normalized_action,
            qty=qty,
        )
        order = self.broker_client.wait_for_order(initial_order.order_id)
        self.sync_from_broker()

        payload = {
            "action": normalized_action,
            "symbol": symbol,
            "qty": qty,
            "price": (
                order.filled_avg_price
                if order.filled_avg_price is not None
                else float(price)
            ),
            "timestamp": timestamp,
            "status": order.status,
            "order_id": order.order_id,
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price,
            "submitted_at": order.submitted_at,
            "filled_at": order.filled_at,
            "broker_provider": self.provider,
        }
        if extra:
            payload.update(extra)
        return _execution_status_from_order(order.status), payload
