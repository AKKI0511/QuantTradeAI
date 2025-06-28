import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import yaml
import pandas as pd
import sys

# Add src to Python path to allow direct import of DataProcessor
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from data.processor import (
    DataProcessor,
)  # Assuming DataProcessor is in src/data/processor.py

# Mock pandas_ta functions if a full import causes issues during testing.
# Specific functions like bbands and sma are mocked within test methods.

# Default parameters from DataProcessor for comparison
DEFAULT_SMA_PERIODS = [5, 10, 20, 50, 200]
DEFAULT_EMA_PERIODS = [5, 10, 20, 50, 200]
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_PARAMS = {"fast": 12, "slow": 26, "signal": 9}
DEFAULT_STOCH_PARAMS = {"k": 14, "d": 3}
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2
DEFAULT_VOLUME_SMA_PERIODS = [5, 10, 20]
DEFAULT_VOLUME_EMA_PERIODS = [5, 10, 20]

DUMMY_CONFIG_PATH = "config/features_config.yaml"  # Relative to repo root


class TestConfigLoading(unittest.TestCase):

    def setUp(self):
        # Create a dummy config file for testing
        self.test_config_dir = "config"  # Should match DUMMY_CONFIG_PATH
        os.makedirs(self.test_config_dir, exist_ok=True)

        self.dummy_config_data = {
            "volatility_features": {"bollinger_bands": {"period": 30, "std_dev": 3}},
            "volume_features": {
                "volume_sma": {"periods": [7, 14, 28]},
                "volume_ema": {"periods": [8, 15, 30]},  # Added for completeness
            },
            "pipeline": {"steps": ["generate_technical_indicators"]},
            # Exclude price_features and momentum_features
            # to test fallback to defaults
        }
        with open(DUMMY_CONFIG_PATH, "w") as f:
            yaml.dump(self.dummy_config_data, f)

    def tearDown(self):
        # Remove the dummy config file created during setUp
        if os.path.exists(DUMMY_CONFIG_PATH):
            os.remove(DUMMY_CONFIG_PATH)
        # Do not remove the 'config' directory itself, as it's a source directory.

    def test_load_parameters_from_config(self):
        processor = DataProcessor()
        self.assertEqual(processor.bb_period, 30)
        self.assertEqual(processor.bb_std, 3)
        self.assertEqual(processor.volume_sma_periods, [7, 14, 28])
        self.assertEqual(processor.volume_ema_periods, [8, 15, 30])
        # Test that parameters not in dummy config fall back to defaults
        self.assertEqual(processor.sma_periods, DEFAULT_SMA_PERIODS)
        self.assertEqual(processor.rsi_period, DEFAULT_RSI_PERIOD)

    @patch("builtins.open", new_callable=mock_open)
    def test_load_parameters_file_not_found(self, mocked_open):
        mocked_open.side_effect = FileNotFoundError

        # Ensure the dummy config is not actually influencing this test
        if os.path.exists(DUMMY_CONFIG_PATH):
            os.remove(DUMMY_CONFIG_PATH)

        processor = DataProcessor()

        self.assertEqual(processor.bb_period, DEFAULT_BB_PERIOD)
        self.assertEqual(processor.bb_std, DEFAULT_BB_STD)
        self.assertEqual(processor.volume_sma_periods, DEFAULT_VOLUME_SMA_PERIODS)
        self.assertEqual(processor.volume_ema_periods, DEFAULT_VOLUME_EMA_PERIODS)
        self.assertEqual(processor.sma_periods, DEFAULT_SMA_PERIODS)
        self.assertEqual(processor.rsi_period, DEFAULT_RSI_PERIOD)

    def test_load_parameters_malformed_config(self):
        # Write invalid YAML to the dummy config file
        with open(DUMMY_CONFIG_PATH, "w") as f:
            f.write("volatility_features: [this is not valid yaml")

        processor = DataProcessor()

        # Assert that all parameters fall back to defaults
        self.assertEqual(processor.bb_period, DEFAULT_BB_PERIOD)
        self.assertEqual(processor.bb_std, DEFAULT_BB_STD)
        self.assertEqual(processor.volume_sma_periods, DEFAULT_VOLUME_SMA_PERIODS)
        self.assertEqual(processor.volume_ema_periods, DEFAULT_VOLUME_EMA_PERIODS)
        self.assertEqual(processor.sma_periods, DEFAULT_SMA_PERIODS)
        self.assertEqual(processor.rsi_period, DEFAULT_RSI_PERIOD)

    def test_load_parameters_list_style_config(self):
        list_style_config = {
            "volatility_features": [{"bollinger_bands": {"period": 30, "std_dev": 3}}],
            "volume_features": [
                {"volume_sma": {"periods": [7, 14, 28]}},
                {"volume_ema": {"periods": [8, 15, 30]}},
            ],
            "price_features": ["close_to_open", "close_to_high"],
            "momentum_features": ["macd"],
            "pipeline": {"steps": ["generate_technical_indicators"]},
        }
        with open(DUMMY_CONFIG_PATH, "w") as f:
            yaml.dump(list_style_config, f)

        processor = DataProcessor()
        self.assertEqual(processor.bb_period, 30)
        self.assertEqual(processor.bb_std, 3)
        self.assertEqual(processor.volume_sma_periods, [7, 14, 28])
        self.assertEqual(processor.volume_ema_periods, [8, 15, 30])


class TestFeatureGenerationMethods(unittest.TestCase):

    def setUp(self):
        # Create a simple dummy DataFrame for testing
        self.dummy_data = {
            "Open": [10, 11, 12, 13, 14],
            "High": [11, 12, 13, 14, 15],
            "Low": [9, 10, 11, 12, 13],
            "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "Volume": [100, 110, 120, 130, 140],
        }
        self.dummy_df = pd.DataFrame(self.dummy_data)

        # Ensure no dummy config file from TestConfigLoading interferes here.
        # DataProcessor will initialize with its defaults if config/features_config.yaml
        # (the real one) doesn't exist or is unparseable during tests,
        # or if patched (like in TestConfigLoading.test_load_parameters_file_not_found).
        # For feature generation tests, we typically want a clean slate or
        # direct override.
        if os.path.exists(
            DUMMY_CONFIG_PATH
        ):  # DUMMY_CONFIG_PATH is 'config/features_config.yaml'
            os.remove(DUMMY_CONFIG_PATH)

    @patch("pandas_ta.bbands")
    def test_add_bollinger_bands_uses_config_params(self, mocked_bbands):
        # Mock the return value of bbands to avoid issues with its internal structure
        # It should return a DataFrame with expected column names for BBANDS
        mock_bb_df = pd.DataFrame(
            {
                "BBL_25_2.5": [1, 2, 3, 4, 5],  # Lower band
                "BBM_25_2.5": [1, 2, 3, 4, 5],  # Middle band
                "BBU_25_2.5": [1, 2, 3, 4, 5],  # Upper band
            }
        )
        mocked_bbands.return_value = mock_bb_df

        processor = DataProcessor()  # Initialize with defaults (no config file)
        processor.bb_period = 25  # Override for this test
        processor.bb_std = 2.5  # Override for this test

        processed_df = processor._add_bollinger_bands(self.dummy_df.copy())

        mocked_bbands.assert_called_once()
        args, kwargs = mocked_bbands.call_args
        self.assertTrue(
            args[0].equals(self.dummy_df["Close"])
        )  # Check the 'close' series
        self.assertEqual(kwargs["length"], 25)
        self.assertEqual(kwargs["std"], 2.5)

        # Check if columns are added
        self.assertIn("bb_lower", processed_df.columns)
        self.assertIn("bb_middle", processed_df.columns)
        self.assertIn("bb_upper", processed_df.columns)

    @patch("pandas_ta.sma")
    def test_add_volume_features_uses_config_params(self, mocked_sma):
        # Mock pandas_ta.sma to return a series, as expected
        mocked_sma.return_value = pd.Series([1.0] * len(self.dummy_df))

        processor = DataProcessor()  # Initialize with defaults
        processor.volume_sma_periods = [8, 15]  # Override for this test
        # Ensure other volume periods are defaults if not testing them or provide mocks
        processor.volume_ema_periods = []

        processed_df = processor._add_volume_features(self.dummy_df.copy())

        # Check calls to sma for volume
        calls = mocked_sma.call_args_list
        self.assertEqual(len(calls), 2)  # Called for each period in volume_sma_periods

        # Check first call (period 8)
        args_0, kwargs_0 = calls[0]
        self.assertTrue(args_0[0].equals(self.dummy_df["Volume"]))
        self.assertEqual(kwargs_0["length"], 8)

        # Check second call (period 15)
        args_1, kwargs_1 = calls[1]
        self.assertTrue(args_1[0].equals(self.dummy_df["Volume"]))
        self.assertEqual(kwargs_1["length"], 15)

        # Check if columns are added (example for one period)
        self.assertIn("volume_sma_8", processed_df.columns)
        self.assertIn("volume_sma_ratio_8", processed_df.columns)
        self.assertIn("volume_sma_15", processed_df.columns)
        self.assertIn("volume_sma_ratio_15", processed_df.columns)
        self.assertIn("obv", processed_df.columns)  # OBV is also added


class TestPipelineExecution(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Open": [1, 2],
                "High": [2, 3],
                "Low": [1, 1.5],
                "Close": [1.5, 2.5],
                "Volume": [100, 110],
            }
        )

    def test_process_data_runs_pipeline_steps(self):
        processor = DataProcessor()
        processor.pipeline = [
            "generate_technical_indicators",
            "handle_missing_values",
            "scale_features",
        ]

        call_sequence = []

        def record(name):
            def inner(df):
                call_sequence.append(name)
                return df

            return inner

        with patch.object(
            processor, "_generate_technical_indicators", side_effect=record("tech")
        ) as _m_tech, patch.object(
            processor, "_handle_missing_values", side_effect=record("missing")
        ) as _m_missing, patch.object(
            processor, "_scale_features", side_effect=record("scale")
        ) as _m_scale, patch.object(
            processor, "_clean_data", side_effect=record("clean")
        ) as _m_clean:
            processor.process_data(self.df.copy())

        self.assertEqual(call_sequence, ["tech", "missing", "scale", "clean"])


if __name__ == "__main__":
    unittest.main()
