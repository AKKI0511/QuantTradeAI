"""Tests for :mod:`quanttradeai.trading.position_manager`."""

from datetime import datetime, timedelta, UTC

import pytest

from quanttradeai.trading import PositionManager
from quanttradeai.utils.config_schemas import (
    DrawdownProtectionConfig,
    MarketImpactConfig,
    PositionManagerConfig,
    RiskManagementConfig,
)


def test_open_close_reconcile() -> None:
    pm = PositionManager.from_config(PositionManagerConfig())
    pm.cash = 1000
    pm.open_position("AAPL", qty=10, price=10)
    pm.handle_market_data({"symbol": "AAPL", "price": 11, "timestamp": datetime.now(UTC)})
    assert pm.portfolio_value == pytest.approx(900 + 10 * 11)
    closed = pm.close_position("AAPL", price=12)
    assert closed == 10
    rec = pm.reconcile_positions(datetime.now(UTC))
    assert rec["intraday"].get("AAPL", 0) == 0
    assert rec["daily"].get("AAPL", 0) == 0


def test_impact_cost() -> None:
    cfg = PositionManagerConfig(
        impact=MarketImpactConfig(enabled=True, model="linear", alpha=0.1, beta=0.05)
    )
    pm = PositionManager.from_config(cfg)
    pm.open_position("AAPL", qty=100, price=10, adv=1000)
    metrics = pm.execution_metrics()
    assert metrics["trades"] == 1
    assert metrics["total_impact_cost"] > 0


def test_risk_manager_integration() -> None:
    cfg = PositionManagerConfig(
        risk_management=RiskManagementConfig(
            drawdown_protection=DrawdownProtectionConfig(enabled=True, max_drawdown_pct=0.1)
        )
    )
    pm = PositionManager.from_config(cfg)
    pm.cash = 10000
    pm.open_position("AAPL", qty=100, price=100)
    ts = datetime.now(UTC)
    pm.handle_market_data({"symbol": "AAPL", "price": 100, "timestamp": ts})
    assert pm.risk_manager is not None
    assert not pm.risk_manager.should_halt_trading()
    pm.handle_market_data({"symbol": "AAPL", "price": 80, "timestamp": ts + timedelta(minutes=1)})
    assert pm.risk_manager.should_halt_trading()


def test_open_position_respects_risk_guard() -> None:
    cfg = PositionManagerConfig(
        risk_management=RiskManagementConfig(
            drawdown_protection=DrawdownProtectionConfig(enabled=True, max_drawdown_pct=0.1)
        )
    )
    pm = PositionManager.from_config(cfg)
    pm.cash = 1000
    ts = datetime.now(UTC)
    assert pm.risk_manager is not None
    pm.risk_manager.update(1000, ts)
    pm.risk_manager.update(900, ts + timedelta(minutes=1))
    assert pm.risk_manager.should_halt_trading()
    pm.open_position("AAPL", qty=10, price=10, timestamp=ts + timedelta(minutes=2))
    assert pm.cash == 1000
    assert "AAPL" not in pm._positions


def test_position_size_multiplier_applied() -> None:
    cfg = PositionManagerConfig(
        risk_management=RiskManagementConfig(
            drawdown_protection=DrawdownProtectionConfig(enabled=True, max_drawdown_pct=0.1)
        )
    )
    pm = PositionManager.from_config(cfg)
    pm.cash = 1000
    ts = datetime.now(UTC)
    assert pm.risk_manager is not None
    pm.risk_manager.update(1000, ts)
    pm.risk_manager.update(905, ts + timedelta(minutes=1))
    assert pm.risk_manager.get_position_size_multiplier() == 0.5
    pm.open_position("AAPL", qty=10, price=10, timestamp=ts + timedelta(minutes=2))
    assert pm._positions["AAPL"].qty == 5
    assert pm.cash == pytest.approx(950)


def test_close_position_not_scaled_by_multiplier() -> None:
    cfg = PositionManagerConfig(
        risk_management=RiskManagementConfig(
            drawdown_protection=DrawdownProtectionConfig(enabled=True, max_drawdown_pct=0.1)
        )
    )
    pm = PositionManager.from_config(cfg)
    pm.cash = 1000
    ts = datetime.now(UTC)
    assert pm.risk_manager is not None
    pm.risk_manager.update(1000, ts)
    pm.open_position("AAPL", qty=10, price=10, timestamp=ts + timedelta(minutes=1))
    pm.risk_manager.update(905, ts + timedelta(minutes=2))
    assert pm.risk_manager.get_position_size_multiplier() == 0.5
    assert not pm.risk_manager.should_halt_trading()
    closed = pm.close_position("AAPL", price=10, timestamp=ts + timedelta(minutes=3))
    assert closed == 10
    assert "AAPL" not in pm._positions
    assert pm.cash == pytest.approx(1000)


def test_close_position_blocked_when_halted() -> None:
    cfg = PositionManagerConfig(
        risk_management=RiskManagementConfig(
            drawdown_protection=DrawdownProtectionConfig(enabled=True, max_drawdown_pct=0.1)
        )
    )
    pm = PositionManager.from_config(cfg)
    pm.cash = 1000
    pm.open_position("AAPL", qty=10, price=10)
    ts = datetime.now(UTC)
    pm.risk_manager.update(1000, ts)
    pm.risk_manager.update(800, ts + timedelta(minutes=1))
    assert pm.risk_manager is not None
    assert pm.risk_manager.should_halt_trading()
    closed = pm.close_position("AAPL", price=10, timestamp=ts + timedelta(minutes=2))
    assert closed == 0
    assert pm._positions["AAPL"].qty == 10

