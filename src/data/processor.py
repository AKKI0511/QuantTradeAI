import pandas as pd
import numpy as np

import logging
import yaml  # Added for YAML loading
from pydantic import ValidationError

from src.utils.config_schemas import FeaturesConfigSchema

# ---------------------------------------------------------------------------
# pandas_ta relies on the deprecated ``numpy.NaN`` constant which was removed
# in numpy 2.0.  Tests mock ``pandas_ta`` functions but importing the library
# would normally fail under numpy>=2 due to this missing attribute.  To keep the
# import working we provide ``numpy.NaN`` when it's absent before importing
# ``pandas_ta``.
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # pragma: no cover - simple compatibility shim

import pandas_ta as ta  # For efficient technical analysis calculations
from src.features import technical as ft
from src.features import custom as cf

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

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            try:
                FeaturesConfigSchema(**config)
            except ValidationError as exc:
                raise ValueError(f"Invalid feature configuration: {exc}") from exc

            # Price-based features -- handle both mapping and list style configs
            price_cfg = config.get("price_features", {})
            if isinstance(price_cfg, dict):
                self.sma_periods = price_cfg.get("sma_periods", default_sma_periods)
                self.ema_periods = price_cfg.get("ema_periods", default_ema_periods)
            else:  # fallback when provided as a simple list
                self.sma_periods = default_sma_periods
                self.ema_periods = default_ema_periods

            # Momentum features
            momentum_cfg = config.get("momentum_features", {})
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

            # Bollinger Bands parameters
            volatility_cfg = config.get("volatility_features", {})
            bb_config = {}
            if isinstance(volatility_cfg, dict):
                bb_config = volatility_cfg.get("bollinger_bands", {})
            elif isinstance(volatility_cfg, list):
                for item in volatility_cfg:
                    if isinstance(item, dict) and "bollinger_bands" in item:
                        bb_config = item["bollinger_bands"]
                        break
            self.bb_period = (
                bb_config.get("period", default_bb_period)
                if isinstance(bb_config, dict)
                else default_bb_period
            )
            self.bb_std = (
                bb_config.get("std_dev", default_bb_std)
                if isinstance(bb_config, dict)
                else default_bb_std
            )

            # Volume indicator parameters
            volume_cfg = config.get("volume_features", {})
            vol_sma_cfg, vol_ema_cfg = {}, {}
            if isinstance(volume_cfg, dict):
                vol_sma_cfg = volume_cfg.get("volume_sma", {})
                vol_ema_cfg = volume_cfg.get("volume_ema", {})
            elif isinstance(volume_cfg, list):
                for item in volume_cfg:
                    if isinstance(item, dict):
                        if "volume_sma" in item:
                            vol_sma_cfg = item["volume_sma"]
                        if "volume_ema" in item:
                            vol_ema_cfg = item["volume_ema"]
            self.volume_sma_periods = vol_sma_cfg.get(
                "periods", default_volume_sma_periods
            )
            self.volume_ema_periods = vol_ema_cfg.get(
                "periods", default_volume_ema_periods
            )

            # Feature combinations for cross/ratio indicators
            comb_cfg = config.get("feature_combinations", [])
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

            # Feature preprocessing and selection
            prep_cfg = config.get("preprocessing", {})
            scale_cfg = (
                prep_cfg.get("scaling", {}) if isinstance(prep_cfg, dict) else {}
            )
            self.scaling_method = scale_cfg.get("method", "standard")
            self.scaling_range = scale_cfg.get("target_range", [-1, 1])

            out_cfg = prep_cfg.get("outliers", {}) if isinstance(prep_cfg, dict) else {}
            self.outlier_method = out_cfg.get("method", "winsorize")
            self.outlier_limits = out_cfg.get("limits", [0.01, 0.99])

            fs_cfg = config.get("feature_selection", {})
            self.feature_selection_method = fs_cfg.get("method", "recursive")
            self.n_features = fs_cfg.get("n_features", None)

            pipe_cfg = (
                config.get("pipeline", {}).get("steps")
                if isinstance(config.get("pipeline", {}), dict)
                else None
            )
            self.pipeline = pipe_cfg or [
                "generate_technical_indicators",
                "generate_volume_features",
                "generate_custom_features",
                "handle_missing_values",
                "remove_outliers",
                "scale_features",
                "select_features",
            ]

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
            self.cross_indicators = []
            self.ratio_indicators = []
            self.scaling_method = "standard"
            self.scaling_range = [-1, 1]
            self.outlier_method = "winsorize"
            self.outlier_limits = [0.01, 0.99]
            self.feature_selection_method = "recursive"
            self.n_features = None
            self.pipeline = [
                "generate_technical_indicators",
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
            self.cross_indicators = []
            self.ratio_indicators = []
            self.scaling_method = "standard"
            self.scaling_range = [-1, 1]
            self.outlier_method = "winsorize"
            self.outlier_limits = [0.01, 0.99]
            self.feature_selection_method = "recursive"
            self.n_features = None
            self.pipeline = [
                "generate_technical_indicators",
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
            "generate_volume_features": self._add_volume_features,
            "generate_custom_features": self._generate_custom_features,
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
                df[f"volume_sma_{period}"] = ft.sma(df["Volume"], period)
                df[f"volume_sma_ratio_{period}"] = (
                    df["Volume"] / df[f"volume_sma_{period}"]
                )

            for period in self.volume_ema_periods:
                df[f"volume_ema_{period}"] = ft.ema(df["Volume"], period)
                df[f"volume_ema_ratio_{period}"] = (
                    df["Volume"] / df[f"volume_ema_{period}"]
                )

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

    def _generate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_momentum_indicators(df)
        df = self._add_bollinger_bands(df)
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

    def _generate_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_custom_features(df)
        df = self._generate_feature_combinations(df)
        return df

    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom features unique to our strategy."""
        try:
            df["momentum_score"] = cf.momentum_score(
                df["Close"],
                df["sma_20"],
                df["rsi"],
                df["macd"],
                df["macd_signal"],
            )

            # 2. Volume-Price Trend
            # Measures buying/selling pressure

            df["vpt"] = (
                df["Volume"]
                * ((df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1))
            ).cumsum()

            df["volatility_breakout"] = cf.volatility_breakout(
                df["High"],
                df["Low"],
                df["Close"],
            )

            # Normalize features
            df["momentum_score"] = (
                df["momentum_score"] - df["momentum_score"].mean()
            ) / df["momentum_score"].std()
            df["vpt"] = (df["vpt"] - df["vpt"].mean()) / df["vpt"].std()

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
