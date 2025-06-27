import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import os, sys

# add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from features import technical as ft
from features import custom as cf

class TestTechnicalFunctions(unittest.TestCase):
    @patch('pandas_ta.sma')
    def test_sma_calls_pandas_ta(self, mock_sma):
        series = pd.Series([1,2,3])
        ft.sma(series, 5)
        mock_sma.assert_called_once_with(series, length=5)

    @patch('pandas_ta.macd')
    def test_macd_returns_dataframe(self, mock_macd):
        mock_df = pd.DataFrame({'MACD_12_26_9':[0], 'MACDs_12_26_9':[0], 'MACDh_12_26_9':[0]})
        mock_macd.return_value = mock_df
        res = ft.macd(pd.Series([1,2,3]))
        self.assertIn('macd', res.columns)
        self.assertIn('signal', res.columns)
        self.assertIn('hist', res.columns)

class TestCustomFunctions(unittest.TestCase):
    def test_momentum_score_basic(self):
        close = pd.Series([10,12,11])
        sma = pd.Series([9,11,12])
        rsi = pd.Series([60,40,55])
        macd = pd.Series([0.5,-0.2,0.1])
        signal = pd.Series([0.3,-0.1,0.0])
        result = cf.momentum_score(close, sma, rsi, macd, signal)
        raw = ((close>sma).astype(int)*0.3 + (rsi>50).astype(int)*0.3 + (macd>signal).astype(int)*0.4)
        expected = (raw - raw.mean())/raw.std()
        pd.testing.assert_series_equal(result, expected)

    def test_volatility_breakout_flag(self):
        high = pd.Series([10]*25)
        low = pd.Series([5]*25)
        close = pd.Series([10]*24 + [16])
        res = cf.volatility_breakout(high, low, close, lookback=20, threshold=1)
        self.assertEqual(res.iloc[-1], 1)
        self.assertEqual(res.iloc[:-1].sum(), 0)

if __name__ == '__main__':
    unittest.main()
