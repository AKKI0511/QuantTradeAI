import unittest
from unittest.mock import patch
import pandas as pd
import os
import yaml
import shutil
import tempfile
import sys


# Add src to Python path to allow direct import of DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.loader import DataLoader

class TestDataLoaderCaching(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmpdir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.config_path = os.path.join(self.tmpdir, "config.yaml")
        self._write_config(expiration=10)

        # create cached dataframe
        self.df = pd.DataFrame({
            'Open': [1.0],
            'High': [2.0],
            'Low': [1.0],
            'Close': [2.0],
            'Volume': [100]
        }, index=pd.date_range('2020-01-01', periods=1))
        self.df.to_parquet(os.path.join(self.cache_dir, 'TEST_data.parquet'))

    def _write_config(self, expiration):
        config = {
            'data': {
                'symbols': ['TEST'],
                'start_date': '2020-01-01',
                'end_date': '2020-01-10',
                'cache_path': self.cache_dir,
                'cache_expiration_days': expiration,
                'use_cache': True,
                'refresh': False
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch('yfinance.Ticker')
    def test_fetch_data_uses_cache(self, mock_ticker):
        loader = DataLoader(self.config_path)
        data_dict = loader.fetch_data()
        mock_ticker.assert_not_called()
        self.assertIn('TEST', data_dict)
        pd.testing.assert_frame_equal(data_dict['TEST'], self.df, check_freq=False)

    @patch('yfinance.Ticker')
    def test_fetch_data_refreshes_cache(self, mock_ticker):
        # remove cached file to force fetch
        os.remove(os.path.join(self.cache_dir, 'TEST_data.parquet'))

        mock_history = pd.DataFrame({
            'Open': [1], 'High': [1], 'Low': [1], 'Close': [1], 'Volume': [1]
        }, index=pd.date_range('2020-01-01', periods=1))
        mock_ticker.return_value.history.return_value = mock_history

        loader = DataLoader(self.config_path)
        data_dict = loader.fetch_data(refresh=True)

        mock_ticker.assert_called_once()
        self.assertTrue(os.path.exists(os.path.join(self.cache_dir, 'TEST_data.parquet')))
        pd.testing.assert_frame_equal(data_dict['TEST'], mock_history)

    @patch('yfinance.Ticker')
    def test_fetch_data_expired_cache(self, mock_ticker):
        self._write_config(expiration=0)

        mock_history = pd.DataFrame({
            'Open': [1], 'High': [1], 'Low': [1], 'Close': [1], 'Volume': [1]
        }, index=pd.date_range('2020-01-01', periods=1))
        mock_ticker.return_value.history.return_value = mock_history

        loader = DataLoader(self.config_path)
        data_dict = loader.fetch_data()

        mock_ticker.assert_called_once()
        pd.testing.assert_frame_equal(data_dict['TEST'], mock_history)

if __name__ == '__main__':
    unittest.main()
