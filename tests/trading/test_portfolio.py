import unittest
import pandas as pd

from quanttradeai.trading.portfolio import Portfolio
from quanttradeai.backtest.backtester import simulate_trades


class TestPortfolio(unittest.TestCase):
    def test_position_size_allocation(self):
        portfolio = Portfolio(capital=10000, risk_per_trade=0.02)
        qty = portfolio.allocate("AAPL", price=100, stop_loss_pct=0.05)
        self.assertEqual(qty, 40)
        self.assertAlmostEqual(portfolio.cash, 6000)

    def test_multi_symbol_backtest(self):
        df_a = pd.DataFrame({"Close": [100, 102, 104], "label": [1, 1, 0]})
        df_b = pd.DataFrame({"Close": [200, 198, 202], "label": [0, -1, 0]})
        portfolio = Portfolio(capital=10000, risk_per_trade=0.02)
        result = simulate_trades({"AAA": df_a, "BBB": df_b}, stop_loss_pct=0.05, portfolio=portfolio)
        self.assertIn("equity_curve", result.columns)
        self.assertAlmostEqual(result["equity_curve"].iloc[-1], 1.008, places=3)


if __name__ == "__main__":
    unittest.main()
