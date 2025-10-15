import unittest
from unittest.mock import patch, call
import pandas as pd
import os
import yaml
import shutil
import tempfile

from quanttradeai.data.loader import DataLoader


class TestDataLoaderCaching(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmpdir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.config_path = os.path.join(self.tmpdir, "config.yaml")
        self._write_config(expiration=10)

        # create cached dataframe
        self.df = pd.DataFrame(
            {
                "Open": [1.0],
                "High": [2.0],
                "Low": [1.0],
                "Close": [2.0],
                "Volume": [100],
            },
            index=pd.date_range("2020-01-01", periods=1),
        )
        self.df.to_parquet(os.path.join(self.cache_dir, "TEST_1d_data.parquet"))

    def _write_config(self, expiration, secondary_timeframes=None):
        config = {
            "data": {
                "symbols": ["TEST"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
                "cache_path": self.cache_dir,
                "timeframe": "1d",
                "cache_expiration_days": expiration,
                "use_cache": True,
                "refresh": False,
            }
        }
        if secondary_timeframes is not None:
            config["data"]["secondary_timeframes"] = secondary_timeframes
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("quanttradeai.data.datasource.YFinanceDataSource.fetch")
    def test_fetch_data_uses_cache(self, mock_fetch):
        loader = DataLoader(self.config_path)
        data_dict = loader.fetch_data()
        mock_fetch.assert_not_called()
        self.assertIn("TEST", data_dict)
        pd.testing.assert_frame_equal(data_dict["TEST"], self.df, check_freq=False)

    @patch("quanttradeai.data.datasource.YFinanceDataSource.fetch")
    def test_fetch_data_refreshes_cache(self, mock_fetch):
        # remove cached file to force fetch
        os.remove(os.path.join(self.cache_dir, "TEST_1d_data.parquet"))

        mock_history = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
            index=pd.date_range("2020-01-01", periods=1),
        )
        mock_fetch.return_value = mock_history

        loader = DataLoader(self.config_path)
        data_dict = loader.fetch_data(refresh=True)

        mock_fetch.assert_called_once()
        self.assertTrue(
            os.path.exists(os.path.join(self.cache_dir, "TEST_1d_data.parquet"))
        )
        pd.testing.assert_frame_equal(data_dict["TEST"], mock_history)

    @patch("quanttradeai.data.datasource.YFinanceDataSource.fetch")
    def test_fetch_data_expired_cache(self, mock_fetch):
        self._write_config(expiration=0)

        mock_history = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
            index=pd.date_range("2020-01-01", periods=1),
        )
        mock_fetch.return_value = mock_history

        loader = DataLoader(self.config_path)
        data_dict = loader.fetch_data()

        mock_fetch.assert_called_once()
        pd.testing.assert_frame_equal(data_dict["TEST"], mock_history)

    @patch("quanttradeai.data.datasource.YFinanceDataSource.fetch")
    def test_fetch_data_with_secondary_timeframes(self, mock_fetch):
        self._write_config(expiration=10, secondary_timeframes=["1h"])

        primary_index = pd.date_range("2020-01-01", periods=2, freq="D")
        primary_df = pd.DataFrame(
            {
                "Open": [100.0, 110.0],
                "High": [101.0, 111.0],
                "Low": [99.0, 109.0],
                "Close": [100.5, 110.5],
                "Volume": [1000, 1100],
            },
            index=primary_index,
        )

        hourly_index = pd.date_range("2020-01-01", periods=48, freq="h")
        hourly_df = pd.DataFrame(
            {
                "Open": range(48),
                "High": [value + 1 for value in range(48)],
                "Low": range(48),
                "Close": [value + 0.5 for value in range(48)],
                "Volume": [10] * 48,
            },
            index=hourly_index,
        )

        mock_fetch.side_effect = [primary_df, hourly_df]

        loader = DataLoader(self.config_path)
        data_dict = loader.fetch_data(refresh=True)

        df = data_dict["TEST"]

        expected_secondary = (
            hourly_df.resample("1D")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .rename(
                columns={
                    "Open": "open_1h_first",
                    "High": "high_1h_max",
                    "Low": "low_1h_min",
                    "Close": "close_1h_last",
                    "Volume": "volume_1h_sum",
                }
            )
            .reindex(primary_index)
        )

        for column in expected_secondary.columns:
            assert column in df.columns
            pd.testing.assert_series_equal(
                df[column], expected_secondary[column], check_names=True
            )

        expected_calls = [
            call("TEST", "2020-01-01", "2020-01-10", "1d"),
            call("TEST", "2020-01-01", "2020-01-10", "1h"),
        ]
        assert mock_fetch.call_args_list == expected_calls


if __name__ == "__main__":
    unittest.main()
