import unittest
import pandas as pd
import os
import yaml
import shutil
import tempfile
import sys
import time

# Add src to Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
)

from data.loader import DataLoader  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
