import unittest
from datetime import datetime, timedelta

from quanttradeai.trading.drawdown_guard import DrawdownGuard
from quanttradeai.trading.risk_manager import RiskManager
from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.utils.config_schemas import DrawdownProtectionConfig, TurnoverLimitsConfig


class TestDrawdownGuard(unittest.TestCase):
    def test_soft_and_hard_stop(self):
        cfg = DrawdownProtectionConfig(
            enabled=True, max_drawdown_pct=0.10, emergency_stop_threshold=1.5
        )
        guard = DrawdownGuard(cfg)
        now = datetime.utcnow()
        guard.update_portfolio_value(100000, now)
        guard.update_portfolio_value(90500, now + timedelta(days=1))
        status = guard.check_drawdown_limits()
        self.assertEqual(status["status"], "soft_stop")
        self.assertEqual(guard.get_position_size_multiplier(), 0.5)

        pm = PortfolioManager(100000, risk_manager=RiskManager(guard))
        pm.cash = 90500
        qty = pm.open_position("AAPL", price=100, stop_loss_pct=0.05)
        self.assertGreater(qty, 0)
        self.assertLess(qty, 2000)  # soft stop halves size

        guard.update_portfolio_value(88000, now + timedelta(days=2))
        pm.cash = 69900  # adjust portfolio to reflect drawdown
        status = guard.check_drawdown_limits()
        self.assertEqual(status["status"], "hard_stop")
        self.assertTrue(guard.should_halt_trading())
        qty2 = pm.open_position("MSFT", price=100, stop_loss_pct=0.05)
        self.assertEqual(qty2, 0)

    def test_turnover_limits(self):
        tcfg = TurnoverLimitsConfig(daily_max=200000)
        guard = DrawdownGuard(DrawdownProtectionConfig(enabled=False), tcfg)
        now = datetime.utcnow()
        guard.update_portfolio_value(100000, now)
        guard.record_trade(180000, now)
        self.assertEqual(guard.get_position_size_multiplier(), 0.5)
        guard.record_trade(50000, now + timedelta(hours=1))
        self.assertTrue(guard.should_halt_trading())

    def test_emergency_stop_liquidation(self):
        cfg = DrawdownProtectionConfig(
            enabled=True, max_drawdown_pct=0.10, emergency_stop_threshold=1.1
        )
        guard = DrawdownGuard(cfg)
        now = datetime.utcnow()
        guard.update_portfolio_value(100000, now)
        pm = PortfolioManager(100000, risk_manager=RiskManager(guard))
        qty = pm.open_position("AAPL", price=100, stop_loss_pct=0.05)
        self.assertGreater(qty, 0)
        pm.positions["AAPL"]["price"] = 70
        guard.update_portfolio_value(pm.portfolio_value, now + timedelta(days=1))
        self.assertTrue(guard.should_emergency_liquidate())
        qty2 = pm.open_position("MSFT", price=100, stop_loss_pct=0.05)
        self.assertEqual(qty2, 0)
        self.assertEqual(pm.positions, {})

    def test_config_path_loading(self):
        guard = DrawdownGuard(config_path="config/risk_config.yaml")
        self.assertEqual(guard.config.max_drawdown_pct, 0.15)
        self.assertEqual(guard.turnover_limits.daily_max, 2.0)


if __name__ == "__main__":
    unittest.main()
