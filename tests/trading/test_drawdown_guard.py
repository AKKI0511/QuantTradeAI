import unittest
from datetime import datetime, timedelta

from quanttradeai.trading.drawdown_guard import DrawdownGuard
from quanttradeai.trading.risk_manager import RiskManager
from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.utils.config_schemas import DrawdownProtectionConfig, TurnoverLimitsConfig


class TestDrawdownGuard(unittest.TestCase):
    def test_soft_and_hard_stop(self):
        cfg = DrawdownProtectionConfig(enabled=True, max_drawdown_pct=0.10)
        guard = DrawdownGuard(cfg)
        now = datetime.utcnow()
        guard.update_portfolio_value(100000, now)
        guard.update_portfolio_value(90500, now + timedelta(days=1))
        self.assertEqual(guard.get_position_size_multiplier(), 0.5)

        pm = PortfolioManager(100000, risk_manager=RiskManager(guard))
        pm.cash = 90500
        qty = pm.open_position("AAPL", price=100, stop_loss_pct=0.05)
        self.assertGreater(qty, 0)
        self.assertLess(qty, 2000)  # soft stop halves size

        guard.update_portfolio_value(88000, now + timedelta(days=2))
        pm.cash = 69900  # adjust portfolio to reflect drawdown
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


if __name__ == "__main__":
    unittest.main()
