import unittest
import os
import tempfile
import yaml
import pandas as pd
from unittest.mock import patch

from quanttradeai.backtest.backtester import simulate_trades, compute_metrics
from quanttradeai.data.processor import DataProcessor
from quanttradeai.main import run_pipeline


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Open": [10, 11, 12, 13, 14, 15],
                "High": [11, 12, 13, 14, 15, 16],
                "Low": [9, 10, 11, 12, 13, 14],
                "Close": [10, 11, 12, 13, 14, 15],
                "Volume": [100] * 6,
            }
        )

    def test_end_to_end_backtest_metrics(self):
        processor = DataProcessor()
        processor.pipeline = []
        with patch.object(processor, "_clean_data", lambda x: x):
            processed = processor.process_data(self.df)
            labeled = processor.generate_labels(processed, forward_returns=1, threshold=0.05)
        simulated = simulate_trades(labeled)
        metrics = compute_metrics(simulated)
        self.assertAlmostEqual(metrics["cumulative_return"], 0.4, places=6)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)

    def test_stop_loss_trigger(self):
        df = pd.DataFrame({"Close": [100, 95, 96], "label": [1, 1, 1]})
        res = simulate_trades(df, stop_loss_pct=0.05)
        self.assertEqual(res["label"].tolist(), [1, 0, 0])
        self.assertAlmostEqual(res["equity_curve"].iloc[-1], 0.95)

    def test_invalid_config_file(self):
        cfg = {"data": {"start_date": "2020-01-01"}}
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.write(fd, yaml.dump(cfg).encode())
        os.close(fd)
        with self.assertRaises(ValueError):
            run_pipeline(path)
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
