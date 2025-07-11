import unittest
from unittest.mock import patch
import pandas as pd
import os
import yaml
import shutil
import tempfile
import time

from quanttradeai.data.loader import DataLoader


class TestCheckMissingDates(unittest.TestCase):
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
        self.loader = DataLoader(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_check_missing_dates_detects_gap(self):
        dates = pd.date_range("2020-01-01", "2020-01-07", freq="B")
        df = pd.DataFrame(
            {"Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1}, index=dates
        )
        df = df.drop(pd.Timestamp("2020-01-03"))
        missing = self.loader._check_missing_dates(df)
        self.assertIn(pd.Timestamp("2020-01-03"), missing)
        self.assertEqual(len(missing), 1)


class TestIsCacheValid(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self.tmpdir, "TEST_data.parquet")
        df = pd.DataFrame({"a": [1]}, index=pd.date_range("2020-01-01", periods=1))
        df.to_parquet(self.cache_file)
        self.config_path = os.path.join(self.tmpdir, "config.yaml")
        config = {
            "data": {
                "symbols": ["TEST"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-02",
                "cache_path": self.tmpdir,
                "cache_expiration_days": 1,
                "use_cache": True,
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
        self.loader = DataLoader(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_is_cache_valid_true(self):
        os.utime(self.cache_file, None)
        self.assertTrue(self.loader._is_cache_valid(self.cache_file))

    def test_is_cache_valid_expired(self):
        old = time.time() - 2 * 24 * 3600
        os.utime(self.cache_file, (old, old))
        self.assertFalse(self.loader._is_cache_valid(self.cache_file))


class TestFetchDataParallel(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmpdir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.config_path = os.path.join(self.tmpdir, "config.yaml")
        config = {
            "data": {
                "symbols": ["AAA", "BBB"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-02",
                "cache_path": self.cache_dir,
                "use_cache": False,
                "max_workers": 2,
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("quanttradeai.data.loader.as_completed", side_effect=lambda fs: fs)
    @patch("quanttradeai.data.loader.ThreadPoolExecutor")
    @patch("quanttradeai.data.datasource.YFinanceDataSource.fetch")
    def test_parallel_execution(self, mock_fetch, mock_executor, _mock_ac):
        class DummyFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class DummyExecutor:
            def __init__(self, max_workers=None):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                pass

            def submit(self, fn, *args):
                return DummyFuture(fn(*args))

        mock_executor.return_value = DummyExecutor(max_workers=2)
        mock_history = pd.DataFrame(
            {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
            index=pd.date_range("2020-01-01", periods=1),
        )
        mock_fetch.return_value = mock_history

        loader = DataLoader(self.config_path)
        data = loader.fetch_data()

        mock_executor.assert_called_once_with(max_workers=2)
        self.assertIn("AAA", data)
        self.assertIn("BBB", data)


if __name__ == "__main__":
    unittest.main()
