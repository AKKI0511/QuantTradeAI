import unittest
import os
import sys
import pandas as pd

# add src to path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")),
)

from backtest.backtester import simulate_trades, compute_metrics  # noqa: E402


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

    def test_compute_metrics(self):
        result = simulate_trades(self.df)
        metrics = compute_metrics(result)
        self.assertAlmostEqual(metrics["cumulative_return"], -0.193333, places=6)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)


if __name__ == "__main__":
    unittest.main()
