import unittest
import pandas as pd

from quanttradeai.backtest.backtester import simulate_trades, compute_metrics


class TestBacktester(unittest.TestCase):
    def setUp(self):
        close = [10, 11, 10, 12, 11]
        labels = [1, 0, -1, 1, -1]
        self.df = pd.DataFrame({"Close": close, "label": labels})

    def test_simulate_trades(self):
        result = simulate_trades(self.df)
        expected_returns = pd.Series(
            [0.1, 0.0, -0.2, -0.083333, 0.0],
            name="strategy_return",
        )
        pd.testing.assert_series_equal(
            result["strategy_return"].round(6).reset_index(drop=True),
            expected_returns.round(6),
            check_names=False,
        )
        self.assertIn("equity_curve", result.columns)

    def test_simulate_trades_with_costs(self):
        result = simulate_trades(self.df, transaction_cost=0.01, slippage=0.01)
        self.assertAlmostEqual(result["strategy_return"].iloc[0], 0.08, places=6)
        self.assertAlmostEqual(result["equity_curve"].iloc[-1], 0.723734, places=6)

    def test_compute_metrics(self):
        result = simulate_trades(self.df)
        metrics = compute_metrics(result)
        self.assertAlmostEqual(metrics["cumulative_return"], -0.193333, places=6)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)


if __name__ == "__main__":
    unittest.main()
