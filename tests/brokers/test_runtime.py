from datetime import UTC, datetime

import pytest

from quanttradeai.brokers.base import (
    BrokerAccountSnapshot,
    BrokerOrderResult,
    BrokerPositionSnapshot,
)
from quanttradeai.brokers.runtime import BrokerExecutionRuntime
from quanttradeai.trading.portfolio import PortfolioManager


class _RiskManagerSpy:
    def __init__(self) -> None:
        self.recorded: list[tuple[float, datetime]] = []

    def update(self, portfolio_value: float, timestamp: datetime) -> None:
        return None

    def record_trade(self, notional: float, timestamp: datetime) -> None:
        self.recorded.append((notional, timestamp))

    def should_emergency_liquidate(self) -> bool:
        return False

    def should_halt_trading(self) -> bool:
        return False

    def get_position_size_multiplier(self) -> float:
        return 1.0


class _FakeBrokerClient:
    provider = "alpaca"

    def __init__(self, starting_cash: float = 1000.0) -> None:
        self.cash = starting_cash
        self.qty = 0
        self.market_price = 0.0
        self.avg_entry_price = 0.0
        self.next_order_id = 1
        self.last_order: dict[str, object] | None = None

    def get_account(self) -> BrokerAccountSnapshot:
        equity = self.cash + (self.qty * self.market_price)
        return BrokerAccountSnapshot(
            account_id="acct-runtime",
            cash=self.cash,
            equity=equity,
            buying_power=self.cash,
        )

    def list_positions(self) -> list[BrokerPositionSnapshot]:
        if self.qty <= 0:
            return []
        return [
            BrokerPositionSnapshot(
                symbol="AAPL",
                qty=self.qty,
                market_price=self.market_price,
                avg_entry_price=self.avg_entry_price,
            )
        ]

    def submit_market_order(
        self,
        *,
        symbol: str,
        action: str,
        qty: int,
    ) -> BrokerOrderResult:
        order_id = f"ord-{self.next_order_id}"
        self.next_order_id += 1
        self.last_order = {
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "qty": qty,
        }
        return BrokerOrderResult(
            order_id=order_id,
            symbol=symbol,
            action=action,
            qty=qty,
            status="new",
        )

    def get_order(self, order_id: str) -> BrokerOrderResult:
        raise AssertionError("Orders are resolved via wait_for_order in this test")

    def wait_for_order(
        self,
        order_id: str,
        *,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> BrokerOrderResult:
        assert self.last_order is not None
        qty = int(self.last_order["qty"])
        action = str(self.last_order["action"])
        fill_price = 10.0 if action == "buy" else 11.0
        filled_at = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        if action == "buy":
            self.cash -= qty * fill_price
            self.qty += qty
            self.avg_entry_price = fill_price
            self.market_price = fill_price
        else:
            self.cash += qty * fill_price
            self.qty = 0
            self.avg_entry_price = 0.0
            self.market_price = fill_price
        return BrokerOrderResult(
            order_id=order_id,
            symbol=str(self.last_order["symbol"]),
            action=action,
            qty=qty,
            status="filled",
            filled_qty=qty,
            filled_avg_price=fill_price,
            filled_at=filled_at,
        )


def test_broker_runtime_records_fill_notional_and_realized_pnl():
    risk_manager = _RiskManagerSpy()
    portfolio = PortfolioManager(capital=1000.0, risk_manager=risk_manager)
    runtime = BrokerExecutionRuntime(
        broker_client=_FakeBrokerClient(),
        portfolio=portfolio,
        position_manager=None,
        stop_loss_pct=0.02,
    )
    runtime.start_session()

    _, buy_payload = runtime.execute_action(
        symbol="AAPL",
        action="buy",
        price=10.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    _, sell_payload = runtime.execute_action(
        symbol="AAPL",
        action="sell",
        price=11.0,
        timestamp=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
    )
    runtime.sync_from_broker()

    assert buy_payload is not None
    assert sell_payload is not None
    assert len(risk_manager.recorded) == 2
    assert risk_manager.recorded[0][0] == pytest.approx(
        buy_payload["filled_qty"] * buy_payload["filled_avg_price"]
    )
    assert risk_manager.recorded[1][0] == pytest.approx(
        sell_payload["filled_qty"] * sell_payload["filled_avg_price"]
    )
    assert portfolio.realized_pnl == pytest.approx(
        sell_payload["filled_qty"]
        * (sell_payload["filled_avg_price"] - buy_payload["filled_avg_price"])
    )
