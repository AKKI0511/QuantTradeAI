import unittest
import os
import pandas as pd

from quanttradeai.backtest.backtester import simulate_trades


class TestRiskManagement(unittest.TestCase):
    def test_stop_loss_closes_trade(self):
        df = pd.DataFrame({"Close": [100, 95, 96], "label": [1, 1, 1]})
        res = simulate_trades(df, stop_loss_pct=0.05)
        self.assertEqual(res["label"].tolist(), [1, 0, 0])
        self.assertAlmostEqual(res["equity_curve"].iloc[-1], 0.95)

    def test_take_profit_closes_trade(self):
        df = pd.DataFrame({"Close": [100, 105, 110, 111], "label": [1, 1, 1, 1]})
        res = simulate_trades(df, take_profit_pct=0.1)
        self.assertEqual(res["label"].tolist(), [1, 1, 0, 0])
        self.assertAlmostEqual(res["equity_curve"].iloc[-1], 1.1)


if __name__ == "__main__":
    unittest.main()
