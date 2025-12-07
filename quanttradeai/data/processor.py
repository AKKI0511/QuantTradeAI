"""Data processing and feature engineering.

The :class:`DataProcessor` transforms raw OHLCV data into a rich feature
set used by models.  It supports a configurable pipeline described in a
YAML file.

Key Components:
    - :class:`DataProcessor`: main entry point for feature generation

Typical Usage:
    ```python
    from quanttradeai.data import DataProcessor
    processor = DataProcessor()
    features = processor.process_data(df)
    ```
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import yaml  # Added for YAML loading
from pydantic import ValidationError
import pandas_ta_classic as ta  # For efficient technical analysis calculations

from quanttradeai.utils.config_schemas import (
    FeaturesConfigSchema,
    MultiTimeframeOperation,
)
from quanttradeai.features import technical as ft
from quanttradeai.features import custom as cf
from quanttradeai.features.sentiment import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw OHLCV data and generate required features for the competition."""

    def __init__(self, config_path: str = "config/features_config.yaml"):
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
        self.sentiment_enabled = False
        self.sentiment_analyzer: SentimentAnalyzer | None = None
        self.price_feature_flags: set[str] = set()
        self.atr_periods: list[int] = []
        self.keltner_periods: list[int] = []
        self.keltner_atr_multiple: float = 2.0
        self.price_momentum_periods: list[int] = []
        self.volume_momentum_periods: list[int] = []
        self.mean_reversion_lookbacks: list[int] = []
        self.volatility_breakout_config: dict[str, Any] | None = None
        self.volume_price_trend_enabled = False
        self.obv_enabled = True

        try:
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f) or {}

            try:
                features_schema = FeaturesConfigSchema(**raw_config)
            except ValidationError as exc:
                raise ValueError(f"Invalid feature configuration: {exc}") from exc

            price_cfg = features_schema.price_features
            self.price_feature_flags = set(price_cfg.enabled)
            self.sma_periods = price_cfg.sma_periods or default_sma_periods
            self.ema_periods = price_cfg.ema_periods or default_ema_periods

            momentum_cfg = raw_config.get("momentum_features", {})
            if isinstance(momentum_cfg, dict):
                self.rsi_period = momentum_cfg.get("rsi_period", default_rsi_period)
                self.macd_params = momentum_cfg.get("macd_params", default_macd_params)
                self.stoch_params = momentum_cfg.get(
                    "stoch_params", default_stoch_params
                )
            else:
                self.rsi_period = default_rsi_period
                self.macd_params = default_macd_params
                self.stoch_params = default_stoch_params

            volatility_cfg = features_schema.volatility_features
            self.atr_periods = list(volatility_cfg.atr_periods)
            self.bb_period = (
                volatility_cfg.bollinger_bands.period
                if volatility_cfg.bollinger_bands
                else default_bb_period
            )
            self.bb_std = (
                volatility_cfg.bollinger_bands.std_dev
                if volatility_cfg.bollinger_bands
                else default_bb_std
            )
            if volatility_cfg.keltner_channels:
                self.keltner_periods = list(volatility_cfg.keltner_channels.periods)
                self.keltner_atr_multiple = volatility_cfg.keltner_channels.atr_multiple

            volume_cfg = features_schema.volume_features
            self.volume_sma_periods = list(volume_cfg.volume_sma.periods)
            self.volume_ema_periods = list(volume_cfg.volume_ema.periods)
            self.volume_sma_ratio_periods = list(volume_cfg.volume_sma_ratios)
            self.volume_ema_ratio_periods = list(volume_cfg.volume_ema_ratios)
            self.volume_price_trend_enabled = volume_cfg.volume_price_trend
            self.obv_enabled = volume_cfg.on_balance_volume

            custom_cfg = features_schema.custom_features
            self.price_momentum_periods = list(custom_cfg.price_momentum)
            self.volume_momentum_periods = list(custom_cfg.volume_momentum)
            self.mean_reversion_lookbacks = list(custom_cfg.mean_reversion)
            self.volatility_breakout_config = (
                custom_cfg.volatility_breakout.model_dump()
                if custom_cfg.volatility_breakout
                else None
            )

            self.multi_timeframe_config = features_schema.multi_timeframe_features
            self.multi_timeframe_enabled = bool(
                self.multi_timeframe_config and self.multi_timeframe_config.enabled
            )
            self.multi_timeframe_operations: list[MultiTimeframeOperation] = (
                list(self.multi_timeframe_config.operations)
                if self.multi_timeframe_config
                else []
            )

            sentiment_cfg = raw_config.get("sentiment", {}) or {}
            self.sentiment_enabled = sentiment_cfg.get("enabled", False)
            if self.sentiment_enabled:
                try:
                    self.sentiment_analyzer = SentimentAnalyzer(
                        provider=sentiment_cfg.get("provider", ""),
                        model=sentiment_cfg.get("model", ""),
                        api_key_env_var=sentiment_cfg.get("api_key_env_var", ""),
                        extra=sentiment_cfg.get("extra", {}),
                    )
                except Exception as exc:
                    logger.error(f"Error initializing sentiment analyzer: {exc}")
                    raise

            comb_cfg = raw_config.get("feature_combinations", [])
            cross_cfg, ratio_cfg = [], []
            if isinstance(comb_cfg, list):
                for item in comb_cfg:
                    if isinstance(item, dict):
                        cross_cfg.extend(item.get("cross_indicators", []))
                        ratio_cfg.extend(item.get("ratio_indicators", []))
            elif isinstance(comb_cfg, dict):
                cross_cfg = comb_cfg.get("cross_indicators", [])
                ratio_cfg = comb_cfg.get("ratio_indicators", [])
            self.cross_indicators = cross_cfg
            self.ratio_indicators = ratio_cfg

            prep_cfg = raw_config.get("preprocessing", {})
            scale_cfg = (
                prep_cfg.get("scaling", {}) if isinstance(prep_cfg, dict) else {}
            )
            self.scaling_method = scale_cfg.get("method", "standard")
            self.scaling_range = scale_cfg.get("target_range", [-1, 1])

            out_cfg = prep_cfg.get("outliers", {}) if isinstance(prep_cfg, dict) else {}
            self.outlier_method = out_cfg.get("method", "winsorize")
            self.outlier_limits = out_cfg.get("limits", [0.01, 0.99])

            fs_cfg = raw_config.get("feature_selection", {})
            self.feature_selection_method = fs_cfg.get("method", "recursive")
            self.n_features = fs_cfg.get("n_features", None)

            pipe_cfg = (
                raw_config.get("pipeline", {}).get("steps")
                if isinstance(raw_config.get("pipeline", {}), dict)
                else None
            )
            self.pipeline = pipe_cfg or [
                "generate_technical_indicators",
                "generate_multi_timeframe_features",
                "generate_volume_features",
                "generate_custom_features",
                "handle_missing_values",
                "remove_outliers",
                "scale_features",
                "select_features",
            ]

            if (
                self.multi_timeframe_enabled
                and "generate_multi_timeframe_features" not in self.pipeline
            ):
                insertion_index = 0
                if "generate_technical_indicators" in self.pipeline:
                    insertion_index = (
                        self.pipeline.index("generate_technical_indicators") + 1
                    )
                self.pipeline.insert(
                    insertion_index,
                    "generate_multi_timeframe_features",
                )

            logger.info("Successfully loaded feature parameters from %s", config_path)

        except FileNotFoundError:
            logger.warning(
                "Feature configuration file %s not found. Using default parameters.",
                config_path,
            )
            self.sma_periods = default_sma_periods
            self.ema_periods = default_ema_periods
            self.rsi_period = default_rsi_period
            self.macd_params = default_macd_params
            self.stoch_params = default_stoch_params
            self.bb_period = default_bb_period
            self.bb_std = default_bb_std
            self.volume_sma_periods = default_volume_sma_periods
            self.volume_ema_periods = default_volume_ema_periods
            self.volume_sma_ratio_periods = default_volume_sma_periods
            self.volume_ema_ratio_periods = default_volume_ema_periods
            self.cross_indicators = []
            self.ratio_indicators = []
            self.scaling_method = "standard"
            self.scaling_range = [-1, 1]
            self.outlier_method = "winsorize"
            self.outlier_limits = [0.01, 0.99]
            self.feature_selection_method = "recursive"
            self.n_features = None
            self.multi_timeframe_config = None
            self.multi_timeframe_enabled = False
            self.multi_timeframe_operations = []
            self.pipeline = [
                "generate_technical_indicators",
                "generate_multi_timeframe_features",
                "generate_volume_features",
                "generate_custom_features",
                "handle_missing_values",
                "remove_outliers",
                "scale_features",
                "select_features",
            ]
        except (yaml.YAMLError, KeyError) as e:
            logger.warning(
                "Error parsing %s or key not found: %s. Using default parameters.",
                config_path,
                e,
            )
            self.sma_periods = default_sma_periods
            self.ema_periods = default_ema_periods
            self.rsi_period = default_rsi_period
            self.macd_params = default_macd_params
            self.stoch_params = default_stoch_params
            self.bb_period = default_bb_period
            self.bb_std = default_bb_std
            self.volume_sma_periods = default_volume_sma_periods
            self.volume_ema_periods = default_volume_ema_periods
            self.volume_sma_ratio_periods = default_volume_sma_periods
            self.volume_ema_ratio_periods = default_volume_ema_periods
            self.cross_indicators = []
            self.ratio_indicators = []
            self.scaling_method = "standard"
            self.scaling_range = [-1, 1]
            self.outlier_method = "winsorize"
            self.outlier_limits = [0.01, 0.99]
            self.feature_selection_method = "recursive"
            self.n_features = None
            self.multi_timeframe_config = None
            self.multi_timeframe_enabled = False
            self.multi_timeframe_operations = []
            self.pipeline = [
                "generate_technical_indicators",
                "generate_multi_timeframe_features",
                "generate_volume_features",
                "generate_custom_features",
                "handle_missing_values",
                "remove_outliers",
                "scale_features",
                "select_features",
            ]

        # Log the actual parameters being used
        logger.info(f"DataProcessor initialized with bb_period: {self.bb_period}")
        logger.info(f"DataProcessor initialized with bb_std: {self.bb_std}")
        logger.info(
            "DataProcessor initialized with volume_sma_periods: "
            f"{self.volume_sma_periods}"
        )
        logger.info(
            "DataProcessor initialized with volume_ema_periods: "
            f"{self.volume_ema_periods}"
        )
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

        step_map = {
            "generate_technical_indicators": self._generate_technical_indicators,
            "generate_multi_timeframe_features": self._generate_multi_timeframe_features,
            "generate_volume_features": self._add_volume_features,
            "generate_custom_features": self._generate_custom_features,
            "generate_sentiment": self._add_sentiment_features,
            "handle_missing_values": self._handle_missing_values,
            "remove_outliers": self._remove_outliers,
            "scale_features": self._scale_features,
            "select_features": self._select_features,
        }

        for step in self.pipeline:
            func = step_map.get(step)
            if func:
                df = func(df)

        df = self._clean_data(df)

        return df

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment score using configured LLM if enabled."""

        if not self.sentiment_enabled or not self.sentiment_analyzer:
            return df
        if "text" not in df.columns:
            logger.warning("Sentiment enabled but 'text' column not found.")
            return df

        try:
            text_series = df["text"]
            valid_mask = text_series.notna()
            if not valid_mask.any():
                logger.warning("Sentiment enabled but no non-null text rows found.")
                return df

            df.loc[valid_mask, "sentiment_score"] = text_series[valid_mask].apply(
                self.sentiment_analyzer.score
            )
        except Exception as exc:
            logger.error(f"Error generating sentiment scores: {exc}")
            raise
        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-derived ratios configured in the YAML file."""

        if not self.price_feature_flags:
            return df

        close = df["Close"]
        open_ = df["Open"]
        high = df["High"]
        low = df["Low"]

        if "close_to_open" in self.price_feature_flags:
            df["close_to_open"] = (close - open_) / open_.replace(0, np.nan)

        if "high_to_low" in self.price_feature_flags:
            df["high_to_low"] = (high - low) / low.replace(0, np.nan)

        if "close_to_high" in self.price_feature_flags:
            df["close_to_high"] = (close - high) / high.replace(0, np.nan)

        if "close_to_low" in self.price_feature_flags:
            df["close_to_low"] = (close - low) / low.replace(0, np.nan)

        if "price_range" in self.price_feature_flags:
            df["price_range"] = (high - low) / close.replace(0, np.nan)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add required momentum indicators."""
        try:
            # SMA
            for period in self.sma_periods:
                df[f"sma_{period}"] = ft.sma(df["Close"], period)

            # EMA
            for period in self.ema_periods:
                df[f"ema_{period}"] = ft.ema(df["Close"], period)

            # RSI
            df["rsi"] = ft.rsi(df["Close"], period=self.rsi_period)

            # MACD
            fast = self.macd_params["fast"]
            slow = self.macd_params["slow"]
            signal = self.macd_params["signal"]
            macd_df = ft.macd(df["Close"], fast=fast, slow=slow, signal=signal)
            df["macd"] = macd_df["macd"]
            df["macd_signal"] = macd_df["signal"]
            df["macd_hist"] = macd_df["hist"]

            # Stochastic Oscillator
            stoch_df = ft.stochastic(
                df["High"],
                df["Low"],
                df["Close"],
                k=self.stoch_params["k"],
                d=self.stoch_params["d"],
            )
            df["stoch_k"] = stoch_df["stoch_k"]
            df["stoch_d"] = stoch_df["stoch_d"]

        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            raise

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands using ``pandas_ta_classic`` and append them."""
        try:
            bb = ta.bbands(df["Close"], length=self.bb_period, std=self.bb_std)
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_middle"] = bb.iloc[:, 1]
            df["bb_upper"] = bb.iloc[:, 2]
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR, Bollinger Bands and Keltner Channels based on configuration."""

        if self.bb_period:
            df = self._add_bollinger_bands(df)

        for period in self.atr_periods:
            df[f"atr_{period}"] = ta.atr(df["High"], df["Low"], df["Close"], length=period)

        for period in self.keltner_periods:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            middle_band = ft.ema(typical_price, period)
            atr = ta.atr(df["High"], df["Low"], df["Close"], length=period)
            df[f"keltner_middle_{period}"] = middle_band
            df[f"keltner_upper_{period}"] = middle_band + self.keltner_atr_multiple * atr
            df[f"keltner_lower_{period}"] = middle_band - self.keltner_atr_multiple * atr

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators such as moving averages and OBV."""
        try:
            for period in self.volume_sma_periods:
                df[f"volume_sma_{period}"] = ft.sma(df["Volume"], period)

            for period in self.volume_sma_ratio_periods:
                if f"volume_sma_{period}" not in df.columns:
                    df[f"volume_sma_{period}"] = ft.sma(df["Volume"], period)
                df[f"volume_sma_ratio_{period}"] = (
                    df["Volume"] / df[f"volume_sma_{period}"]
                )

            for period in self.volume_ema_periods:
                df[f"volume_ema_{period}"] = ft.ema(df["Volume"], period)

            for period in self.volume_ema_ratio_periods:
                if f"volume_ema_{period}" not in df.columns:
                    df[f"volume_ema_{period}"] = ft.ema(df["Volume"], period)
                df[f"volume_ema_ratio_{period}"] = (
                    df["Volume"] / df[f"volume_ema_{period}"]
                )

            if self.obv_enabled:
                df["obv"] = ta.obv(df["Close"], df["Volume"])

            if self.volume_price_trend_enabled:
                df["volume_price_trend"] = (
                    df["Volume"]
                    * (
                        (df["Close"] - df["Close"].shift(1))
                        / df["Close"].shift(1)
                    )
                ).cumsum()
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

    def _generate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_price_features(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_features(df)
        df = self._add_return_features(df)
        return df

    def _generate_feature_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cross and ratio indicator features based on configuration."""
        for pair in self.cross_indicators:
            if len(pair) == 2 and pair[0] in df.columns and pair[1] in df.columns:
                col_name = f"cross_{pair[0]}_{pair[1]}"
                df[col_name] = (
                    (df[pair[0]] > df[pair[1]])
                    & (df[pair[0]].shift(1) <= df[pair[1]].shift(1))
                ).astype(int)

        for pair in self.ratio_indicators:
            if len(pair) == 2 and pair[0] in df.columns and pair[1] in df.columns:
                col_name = f"ratio_{pair[0]}_{pair[1]}"
                with np.errstate(divide="ignore", invalid="ignore"):
                    df[col_name] = df[pair[0]] / df[pair[1]]

        return df

    def _generate_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features that combine secondary timeframe aggregates."""

        if not self.multi_timeframe_enabled or not self.multi_timeframe_operations:
            return df

        primary_column_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        default_stat_map = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        for operation in self.multi_timeframe_operations:
            base = operation.base.lower()
            primary_col = operation.primary_column or primary_column_map.get(base)
            if not primary_col or primary_col not in df.columns:
                logger.debug(
                    "Skipping multi-timeframe feature for base %s: primary column %s missing",
                    base,
                    primary_col,
                )
                continue

            stat = (operation.stat or default_stat_map.get(base) or "")
            candidate_columns = []
            if stat:
                candidate_columns.append(f"{base}_{operation.timeframe}_{stat}")
            candidate_columns.extend(
                col
                for col in df.columns
                if col.startswith(f"{base}_{operation.timeframe}_")
            )

            secondary_column = next(
                (col for col in candidate_columns if col in df.columns),
                None,
            )

            if secondary_column is None:
                logger.debug(
                    "Secondary timeframe column missing for base %s and timeframe %s",
                    base,
                    operation.timeframe,
                )
                continue

            feature_name = operation.feature_name
            if not feature_name:
                feature_name_parts = [
                    "mtf",
                    operation.type,
                    base,
                    operation.timeframe,
                ]
                if stat:
                    feature_name_parts.append(stat)
                if operation.type == "rolling_divergence" and operation.rolling_window:
                    feature_name_parts.append(str(operation.rolling_window))
                feature_name = "_".join(feature_name_parts)

            secondary_series = df[secondary_column]
            primary_series = df[primary_col]

            if operation.type == "ratio":
                with np.errstate(divide="ignore", invalid="ignore"):
                    df[feature_name] = secondary_series / primary_series
            elif operation.type == "delta":
                df[feature_name] = secondary_series - primary_series
            elif operation.type == "pct_change":
                with np.errstate(divide="ignore", invalid="ignore"):
                    df[feature_name] = (
                        (secondary_series - primary_series) / primary_series
                    )
            elif operation.type == "rolling_divergence":
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = secondary_series / primary_series
                window = operation.rolling_window or 5
                df[feature_name] = ratio - ratio.rolling(window).mean()

        return df

    def _generate_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_custom_features(df)
        df = self._generate_feature_combinations(df)
        return df

    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom features unique to our strategy."""
        try:
            for period in self.price_momentum_periods:
                df[f"price_momentum_{period}"] = df["Close"].pct_change(period)

            for period in self.volume_momentum_periods:
                df[f"volume_momentum_{period}"] = df["Volume"].pct_change(period)

            for lookback in self.mean_reversion_lookbacks:
                rolling_mean = df["Close"].rolling(lookback).mean()
                rolling_std = df["Close"].rolling(lookback).std()
                df[f"mean_reversion_{lookback}"] = (
                    df["Close"] - rolling_mean
                ) / rolling_std

            if self.volatility_breakout_config:
                lookbacks = self.volatility_breakout_config.get("lookback") or []
                threshold = self.volatility_breakout_config.get("threshold", 2.0)
                for lookback in lookbacks:
                    column_name = f"volatility_breakout_{lookback}"
                    df[column_name] = cf.volatility_breakout(
                        df["High"],
                        df["Low"],
                        df["Close"],
                        lookback=lookback,
                        threshold=threshold,
                    )
                if lookbacks:
                    df["volatility_breakout"] = df[f"volatility_breakout_{lookbacks[0]}"]

            required_cols = {"sma_20", "rsi", "macd", "macd_signal"}
            if required_cols.issubset(df.columns):
                df["momentum_score"] = cf.momentum_score(
                    df["Close"],
                    df["sma_20"],
                    df["rsi"],
                    df["macd"],
                    df["macd_signal"],
                )

            # Keep features in native scale; normalization handled by scaler

        except Exception as e:
            logger.error(f"Error calculating custom features: {str(e)}")
            raise

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using forward/backward fill."""
        return df.ffill().bfill()

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or clip outliers based on configuration."""
        lower, upper = self.outlier_limits
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if self.outlier_method == "winsorize":
            for col in numeric_cols:
                q_low = df[col].quantile(lower)
                q_high = df[col].quantile(upper)
                df[col] = df[col].clip(q_low, q_high)
        elif self.outlier_method == "clip":
            for col in numeric_cols:
                q_low = df[col].quantile(lower)
                q_high = df[col].quantile(upper)
                df[col] = df[col].clip(q_low, q_high)
        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using sklearn scalers."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = None
        if self.scaling_method == "minmax":
            scaler = MinMaxScaler(feature_range=tuple(self.scaling_range))
        elif self.scaling_method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple feature selection keeping the first n features if specified."""
        if self.n_features is None:
            return df
        selected_cols = list(df.columns[: self.n_features])
        return df[selected_cols]

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
