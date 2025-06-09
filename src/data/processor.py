import pandas as pd
import numpy as np
from typing import Dict, List, Union
import logging
import pandas_ta as ta  # For efficient technical analysis calculations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw OHLCV data and generate required features for the competition."""

    def __init__(self):
        """Initialize DataProcessor with default parameters."""
        # Parameters for technical indicators
        self.sma_periods = [5, 10, 20, 50, 200]
        self.ema_periods = [5, 10, 20, 50, 200]
        self.rsi_period = 14
        self.macd_params = {"fast": 12, "slow": 26, "signal": 9}
        self.stoch_params = {"k": 14, "d": 3}

        # Bollinger Bands parameters
        self.bb_period = 20
        self.bb_std = 2

        # Volume indicator parameters
        self.volume_sma_periods = [5, 10, 20]
        self.volume_ema_periods = [5, 10, 20]

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
