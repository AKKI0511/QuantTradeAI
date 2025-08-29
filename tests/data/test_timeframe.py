import unittest
from unittest.mock import patch
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

if __name__ == "__main__":
    unittest.main()
