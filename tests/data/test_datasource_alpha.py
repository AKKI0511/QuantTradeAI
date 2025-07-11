import unittest
from unittest.mock import patch
import pandas as pd
import os
import yaml
import shutil
import tempfile

from quanttradeai.data.loader import DataLoader
from quanttradeai.data.datasource import AlphaVantageDataSource


class TestAlphaVantageAdapter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "config.yaml")
        config = {
            "data": {
                "symbols": ["TEST"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
                "cache_path": self.tmpdir,
                "use_cache": False,
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("quanttradeai.data.datasource.AlphaVantageDataSource.fetch")
    def test_fetch_with_alpha_vantage(self, mock_fetch):
        df = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
            index=pd.date_range("2020-01-01", periods=1),
        )
        mock_fetch.return_value = df
        source = AlphaVantageDataSource(api_key="demo")
        loader = DataLoader(self.config_path, data_source=source)
        data = loader.fetch_data(refresh=True)
        mock_fetch.assert_called_once_with("TEST", "2020-01-01", "2020-01-10")
        pd.testing.assert_frame_equal(data["TEST"], df)


if __name__ == "__main__":
    unittest.main()

