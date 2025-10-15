import unittest
from unittest.mock import patch, call
import pandas as pd
import os
import yaml
import tempfile

from quanttradeai.data.loader import DataLoader

class TestDataLoaderTimeframe(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "config.yaml")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    @patch("quanttradeai.data.datasource.YFinanceDataSource.fetch")
    def test_custom_timeframe_passed(self, mock_fetch):
        config = {
            "data": {
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-02",
                "cache_path": self.tmpdir,
                "use_cache": False,
                "timeframe": "1h",
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        mock_fetch.return_value = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
            index=pd.date_range("2020-01-01", periods=1, freq="h"),
        )
        loader = DataLoader(self.config_path)
        loader.fetch_data()
        mock_fetch.assert_called_once_with("AAA", "2020-01-01", "2020-01-02", "1h")

    @patch("quanttradeai.data.datasource.YFinanceDataSource.fetch")
    def test_default_timeframe(self, mock_fetch):
        config = {
            "data": {
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-02",
                "cache_path": self.tmpdir,
                "use_cache": False,
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        mock_fetch.return_value = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
            index=pd.date_range("2020-01-01", periods=1),
        )
        loader = DataLoader(self.config_path)
        loader.fetch_data()
        mock_fetch.assert_called_once_with("AAA", "2020-01-01", "2020-01-02", "1d")

    @patch("quanttradeai.data.datasource.YFinanceDataSource.fetch")
    def test_secondary_timeframes_calls_fetch(self, mock_fetch):
        config = {
            "data": {
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-03",
                "cache_path": self.tmpdir,
                "use_cache": False,
                "timeframe": "1d",
                "secondary_timeframes": ["1h", "30m"],
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        primary_index = pd.date_range("2020-01-01", periods=2, freq="D")
        primary_df = pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
                "Volume": [100, 110],
            },
            index=primary_index,
        )

        hourly_df = pd.DataFrame(
            {
                "Open": range(48),
                "High": [value + 1 for value in range(48)],
                "Low": range(48),
                "Close": [value + 0.5 for value in range(48)],
                "Volume": [5] * 48,
            },
            index=pd.date_range("2020-01-01", periods=48, freq="h"),
        )

        thirty_df = pd.DataFrame(
            {
                "Open": range(96),
                "High": [value + 1 for value in range(96)],
                "Low": range(96),
                "Close": [value + 0.25 for value in range(96)],
                "Volume": [2] * 96,
            },
            index=pd.date_range("2020-01-01", periods=96, freq="30min"),
        )

        mock_fetch.side_effect = [primary_df, hourly_df, thirty_df]

        loader = DataLoader(self.config_path)
        data = loader.fetch_data()

        expected_calls = [
            call("AAA", "2020-01-01", "2020-01-03", "1d"),
            call("AAA", "2020-01-01", "2020-01-03", "1h"),
            call("AAA", "2020-01-01", "2020-01-03", "30m"),
        ]
        assert mock_fetch.call_args_list == expected_calls

        df = data["AAA"]
        assert "close_1h_last" in df.columns
        assert "volume_30m_sum" in df.columns

if __name__ == "__main__":
    unittest.main()
