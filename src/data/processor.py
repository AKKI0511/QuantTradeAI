import pandas as pd
import numpy as np
from typing import Dict, List, Union
import logging
import yaml  # Added for YAML loading

# ---------------------------------------------------------------------------
# pandas_ta relies on the deprecated ``numpy.NaN`` constant which was removed
# in numpy 2.0.  Tests mock ``pandas_ta`` functions but importing the library
# would normally fail under numpy>=2 due to this missing attribute.  To keep the
# import working we provide ``numpy.NaN`` when it's absent before importing
# ``pandas_ta``.
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # pragma: no cover - simple compatibility shim

import pandas_ta as ta  # For efficient technical analysis calculations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw OHLCV data and generate required features for the competition."""

    def __init__(self):
        """Initialize DataProcessor with parameters from config or defaults."""
        # Default parameters for technical indicators
        default_sma_periods = [5, 10, 20, 50, 200]
        default_ema_periods = [5, 10, 20, 50, 200]
        default_rsi_period = 14
        default_macd_params = {"fast": 12, "slow": 26, "signal": 9}
        default_stoch_params = {"k": 14, "d": 3}
        default_bb_period = 20
        default_bb_std = 2
        default_volume_sma_periods = [5, 10, 20]
        default_volume_ema_periods = [5, 10, 20]

        config_path = "config/features_config.yaml"
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Price-based features -- handle both mapping and list style configs
            price_cfg = config.get('price_features', {})
            if isinstance(price_cfg, dict):
                self.sma_periods = price_cfg.get('sma_periods', default_sma_periods)
                self.ema_periods = price_cfg.get('ema_periods', default_ema_periods)
            else:  # fallback when provided as a simple list
                self.sma_periods = default_sma_periods
                self.ema_periods = default_ema_periods

            # Momentum features
            momentum_cfg = config.get('momentum_features', {})
            if isinstance(momentum_cfg, dict):
                self.rsi_period = momentum_cfg.get('rsi_period', default_rsi_period)
                self.macd_params = momentum_cfg.get('macd_params', default_macd_params)
                self.stoch_params = momentum_cfg.get('stoch_params', default_stoch_params)
            else:
                self.rsi_period = default_rsi_period
                self.macd_params = default_macd_params
                self.stoch_params = default_stoch_params

            # Bollinger Bands parameters
            volatility_cfg = config.get('volatility_features', {})
            bb_config = {}
            if isinstance(volatility_cfg, dict):
                bb_config = volatility_cfg.get('bollinger_bands', {})
            elif isinstance(volatility_cfg, list):
                for item in volatility_cfg:
                    if isinstance(item, dict) and 'bollinger_bands' in item:
                        bb_config = item['bollinger_bands']
                        break
            self.bb_period = bb_config.get('period', default_bb_period) if isinstance(bb_config, dict) else default_bb_period
            self.bb_std = bb_config.get('std_dev', default_bb_std) if isinstance(bb_config, dict) else default_bb_std

            # Volume indicator parameters
            volume_cfg = config.get('volume_features', {})
            vol_sma_cfg, vol_ema_cfg = {}, {}
            if isinstance(volume_cfg, dict):
                vol_sma_cfg = volume_cfg.get('volume_sma', {})
                vol_ema_cfg = volume_cfg.get('volume_ema', {})
            elif isinstance(volume_cfg, list):
                for item in volume_cfg:
                    if isinstance(item, dict):
                        if 'volume_sma' in item:
                            vol_sma_cfg = item['volume_sma']
                        if 'volume_ema' in item:
                            vol_ema_cfg = item['volume_ema']
            self.volume_sma_periods = vol_sma_cfg.get('periods', default_volume_sma_periods)
            self.volume_ema_periods = vol_ema_cfg.get('periods', default_volume_ema_periods)

            logger.info("Successfully loaded feature parameters from %s", config_path)

        except FileNotFoundError:
            logger.warning("Feature configuration file %s not found. Using default parameters.", config_path)
            self.sma_periods = default_sma_periods
            self.ema_periods = default_ema_periods
            self.rsi_period = default_rsi_period
            self.macd_params = default_macd_params
            self.stoch_params = default_stoch_params
            self.bb_period = default_bb_period
            self.bb_std = default_bb_std
            self.volume_sma_periods = default_volume_sma_periods
            self.volume_ema_periods = default_volume_ema_periods
        except (yaml.YAMLError, KeyError) as e:
            logger.warning("Error parsing %s or key not found: %s. Using default parameters.", config_path, e)
            self.sma_periods = default_sma_periods
            self.ema_periods = default_ema_periods
            self.rsi_period = default_rsi_period
            self.macd_params = default_macd_params
            self.stoch_params = default_stoch_params
            self.bb_period = default_bb_period
            self.bb_std = default_bb_std
            self.volume_sma_periods = default_volume_sma_periods
            self.volume_ema_periods = default_volume_ema_periods

        # Log the actual parameters being used
        logger.info(f"DataProcessor initialized with bb_period: {self.bb_period}")
        logger.info(f"DataProcessor initialized with bb_std: {self.bb_std}")
        logger.info(f"DataProcessor initialized with volume_sma_periods: {self.volume_sma_periods}")
        logger.info(f"DataProcessor initialized with volume_ema_periods: {self.volume_ema_periods}")
        logger.info(f"DataProcessor initialized with sma_periods: {self.sma_periods}")

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw OHLCV data and generate all required features.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with all technical indicators and features
        """
        df = data.copy()

        # 1. Generate Momentum Indicators
        df = self._add_momentum_indicators(df)

        # 2. Generate Bollinger Bands
        df = self._add_bollinger_bands(df)

        # 3. Generate Volume-based Features
        df = self._add_volume_features(df)

        # 4. Generate Return-based Features
        df = self._add_return_features(df)

        # 5. Generate Custom Features
        df = self._add_custom_features(df)

        # 6. Clean up and validate
        df = self._clean_data(df)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add required momentum indicators."""
        try:
            # SMA
            for period in self.sma_periods:
                df[f"sma_{period}"] = ta.sma(df["Close"], length=period)

            # EMA
            for period in self.ema_periods:
                df[f"ema_{period}"] = ta.ema(df["Close"], length=period)

            # RSI
            df["rsi"] = ta.rsi(df["Close"], length=self.rsi_period)

            # MACD
            fast = self.macd_params["fast"]
            slow = self.macd_params["slow"]
            signal = self.macd_params["signal"]
            macd = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)

            macd_col = f"MACD_{fast}_{slow}_{signal}"
            macds_col = f"MACDs_{fast}_{slow}_{signal}"
            macdh_col = f"MACDh_{fast}_{slow}_{signal}"

            df["macd"] = macd[macd_col]
            df["macd_signal"] = macd[macds_col]
            df["macd_hist"] = macd[macdh_col]

            # Stochastic Oscillator
            stoch = ta.stoch(
                df["High"],
                df["Low"],
                df["Close"],
                k=self.stoch_params["k"],
                d=self.stoch_params["d"],
            )
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]

        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            raise

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands using ``pandas_ta`` and append them."""
        try:
            bb = ta.bbands(df["Close"], length=self.bb_period, std=self.bb_std)
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_middle"] = bb.iloc[:, 1]
            df["bb_upper"] = bb.iloc[:, 2]
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators such as moving averages and OBV."""
        try:
            for period in self.volume_sma_periods:
                df[f"volume_sma_{period}"] = ta.sma(df["Volume"], length=period)
                df[f"volume_sma_ratio_{period}"] = df["Volume"] / df[f"volume_sma_{period}"]

            for period in self.volume_ema_periods:
                df[f"volume_ema_{period}"] = ta.ema(df["Volume"], length=period)
                df[f"volume_ema_ratio_{period}"] = df["Volume"] / df[f"volume_ema_{period}"]

            df["obv"] = ta.obv(df["Close"], df["Volume"])
        except Exception as e:
            logger.error(f"Error calculating volume features: {str(e)}")
            raise

        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        try:
            # Daily returns
            df["daily_return"] = df["Close"].pct_change()

            # Weekly returns (5 trading days)
            df["weekly_return"] = df["Close"].pct_change(5)

            # Monthly returns (21 trading days)
            df["monthly_return"] = df["Close"].pct_change(21)

            # Rolling volatility
            df["volatility_21d"] = df["daily_return"].rolling(21).std()

        except Exception as e:
            logger.error(f"Error calculating return features: {str(e)}")
            raise

        return df

    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom features unique to our strategy."""
        try:
            # 1. Price Momentum Score
            # Combines multiple momentum indicators into a single score
            df["momentum_score"] = (
                (df["Close"] > df["sma_20"]).astype(int) * 0.3
                + (df["rsi"] > 50).astype(int) * 0.3
                + (df["macd"] > df["macd_signal"]).astype(int) * 0.4
            )

            # 2. Volume-Price Trend
            # Measures buying/selling pressure
            df["vpt"] = (
                df["Volume"]
                * ((df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1))
            ).cumsum()

            # Normalize features
            df["momentum_score"] = (
                df["momentum_score"] - df["momentum_score"].mean()
            ) / df["momentum_score"].std()
            df["vpt"] = (df["vpt"] - df["vpt"].mean()) / df["vpt"].std()

        except Exception as e:
            logger.error(f"Error calculating custom features: {str(e)}")
            raise

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate processed data."""
        # Remove first 200 rows due to NaN values from long-term indicators
        df = df.iloc[200:]

        # Forward fill any remaining NaN values
        df = df.ffill()

        # Remove any remaining NaN values
        df = df.dropna()

        return df

    def generate_labels(
        self, df: pd.DataFrame, forward_returns: int = 5, threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Generate trading signals based on forward returns.

        Args:
            df: DataFrame with features
            forward_returns: Number of days to look ahead
            threshold: Return threshold for buy/sell signals

        Returns:
            DataFrame with added labels column
        """
        # Calculate forward returns
        df["forward_returns"] = df["Close"].shift(-forward_returns) / df["Close"] - 1

        # Generate labels: 1 (buy), 0 (hold), -1 (sell)
        df["label"] = 0
        df.loc[df["forward_returns"] > threshold, "label"] = 1
        df.loc[df["forward_returns"] < -threshold, "label"] = -1

        # Remove last few rows where we can't calculate forward returns
        df = df[:-forward_returns]

        return df
