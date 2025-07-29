import unittest
import pandas as pd

from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.trading.risk import position_size
from quanttradeai.backtest.backtester import simulate_trades


class TestPortfolioManager(unittest.TestCase):
    def test_position_size_used(self):
        pm = PortfolioManager(10000, max_risk_per_trade=0.02, max_portfolio_risk=0.1)
        qty = pm.open_position("AAPL", price=100, stop_loss_pct=0.05)
        self.assertEqual(qty, position_size(10000, 0.02, 0.05, 100))
        self.assertAlmostEqual(pm.risk_exposure, qty * 100 * 0.05 / pm.portfolio_value)

    def test_portfolio_risk_cap(self):
        pm = PortfolioManager(10000, max_risk_per_trade=0.04, max_portfolio_risk=0.05)
        pm.open_position("AAPL", 100, 0.05)
        qty = pm.open_position("TSLA", 200, 0.05)
        self.assertLessEqual(pm.risk_exposure, 0.05 + 1e-6)
        self.assertGreaterEqual(qty, 0)

    def test_open_position_without_stop_loss(self):
        pm = PortfolioManager(10000, max_risk_per_trade=0.02)
        qty = pm.open_position("AAPL", price=50, stop_loss_pct=None)
        expected_qty = int(pm.max_risk_per_trade * pm.initial_capital / 50)
        self.assertEqual(qty, expected_qty)
        self.assertEqual(pm.positions["AAPL"]["stop_loss_pct"], 0.0)
        self.assertEqual(pm.risk_exposure, 0.0)

    def test_multi_symbol_simulation(self):
        df1 = pd.DataFrame({"Close": [100, 101, 102], "label": [1, 0, 0]})
        df2 = pd.DataFrame({"Close": [50, 49, 48], "label": [-1, 0, 0]})
        pm = PortfolioManager(10000)
        res = simulate_trades({"s1": df1, "s2": df2}, portfolio=pm, stop_loss_pct=0.05)
        self.assertIn("portfolio", res)
        self.assertIn("equity_curve", res["portfolio"].columns)


if __name__ == "__main__":
    unittest.main()
