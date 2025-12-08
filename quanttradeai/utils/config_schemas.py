"""Pydantic configuration schemas.

Defines validation models for data and feature configuration files used
throughout the project.

Key Components:
    - :class:`DataSection`
    - :class:`ModelConfigSchema`
    - :class:`FeaturesConfigSchema`
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal, Set

import yaml
from pydantic import BaseModel, Field, model_validator


IMPACT_CONFIG_PATH = Path("config/impact_config.yaml")


def _known_asset_classes(config_path: Path = IMPACT_CONFIG_PATH) -> set[str]:
    if not config_path.is_file():
        return set()

    try:
        with config_path.open("r") as file:
            cfg = yaml.safe_load(file) or {}
    except yaml.YAMLError:  # pragma: no cover - defensive guard
        return set()

    asset_classes = cfg.get("asset_classes", {})
    if isinstance(asset_classes, dict):
        return set(asset_classes)
    return set()


class DataSection(BaseModel):
    symbols: List[str]
    asset_classes: Dict[str, str] = Field(default_factory=dict)
    start_date: str
    end_date: str
    timeframe: Optional[str] = "1d"
    secondary_timeframes: Optional[List[str]] = None
    cache_path: Optional[str] = None
    cache_dir: Optional[str] = None
    cache_expiration_days: Optional[int] = None
    use_cache: Optional[bool] = True
    refresh: Optional[bool] = False
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    max_workers: Optional[int] = 1

    @model_validator(mode="before")
    @classmethod
    def normalize_symbols(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        raw_symbols = values.get("symbols")
        if raw_symbols is None:
            return values

        if not isinstance(raw_symbols, list):
            raise ValueError("data.symbols must be a list of tickers or mappings.")

        known_assets = _known_asset_classes()
        tickers: list[str] = []
        asset_map: dict[str, str] = {}
        for item in raw_symbols:
            if isinstance(item, str):
                ticker = item
                asset_class = "equities"
            elif isinstance(item, dict):
                ticker = item.get("ticker")
                if not ticker:
                    raise ValueError(
                        "data.symbols entries must include a 'ticker' field when using mapping syntax."
                    )
                asset_class = item.get("asset_class", "equities")
            else:
                raise ValueError(
                    "data.symbols entries must be strings or mappings with 'ticker' and optional 'asset_class'."
                )

            if known_assets and asset_class not in known_assets:
                raise ValueError(
                    f"data.symbols asset_class '{asset_class}' is not defined in config/impact_config.yaml."
                )

            tickers.append(ticker)
            asset_map[ticker] = asset_class

        values["symbols"] = tickers
        values["asset_classes"] = asset_map
        return values

    @staticmethod
    def _parse_date(value: str, field_name: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                f"data.{field_name} must be in ISO format YYYY-MM-DD. Received: {value!r}."
            ) from exc

    @model_validator(mode="after")
    def validate_test_window(self) -> "DataSection":
        start_dt = self._parse_date(self.start_date, "start_date")
        end_dt = self._parse_date(self.end_date, "end_date")
        if start_dt > end_dt:
            raise ValueError("data.start_date must be on or before data.end_date.")

        test_start_dt = None
        if self.test_start is not None:
            test_start_dt = self._parse_date(self.test_start, "test_start")
            if not (start_dt <= test_start_dt <= end_dt):
                raise ValueError(
                    "data.test_start must fall within the configured data.start_date and data.end_date range."
                )

        if self.test_end is not None:
            test_end_dt = self._parse_date(self.test_end, "test_end")
            if not (start_dt <= test_end_dt <= end_dt):
                raise ValueError(
                    "data.test_end must fall within the configured data.start_date and data.end_date range."
                )
            if test_start_dt and test_start_dt > test_end_dt:
                raise ValueError("data.test_start must be on or before data.test_end.")

        return self


class NewsConfig(BaseModel):
    """Configuration for optional news ingestion used by sentiment features."""

    enabled: bool = False
    provider: Optional[str] = "yfinance"
    lookback_days: int = 30
    symbols: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_lookback(self) -> "NewsConfig":
        if self.lookback_days < 0:
            raise ValueError("news.lookback_days must be non-negative")
        return self


class ModelConfigSchema(BaseModel):
    data: DataSection
    news: Optional[NewsConfig] = None


class PipelineConfig(BaseModel):
    steps: List[str]


class SentimentConfig(BaseModel):
    """Configuration for LLM-based sentiment analysis."""

    enabled: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key_env_var: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class MultiTimeframeOperation(BaseModel):
    """Configuration for a derived multi-timeframe feature."""

    type: Literal["ratio", "delta", "pct_change", "rolling_divergence"]
    timeframe: str
    base: Literal["open", "high", "low", "close", "volume"]
    stat: Optional[str] = None
    primary_column: Optional[str] = None
    feature_name: Optional[str] = None
    rolling_window: Optional[int] = Field(default=None, gt=1)

    @model_validator(mode="after")
    def validate_rolling_window(self) -> "MultiTimeframeOperation":
        if self.type == "rolling_divergence" and self.rolling_window is None:
            raise ValueError(
                "multi_timeframe_features.operations rolling_divergence requires a rolling_window value greater than 1."
            )
        return self


class MultiTimeframeConfig(BaseModel):
    enabled: bool = False
    operations: List[MultiTimeframeOperation] = Field(default_factory=list)


class PriceFeaturesConfig(BaseModel):
    enabled: Set[
        Literal[
            "close_to_open",
            "high_to_low",
            "close_to_high",
            "close_to_low",
            "price_range",
        ]
    ] = Field(default_factory=set)
    sma_periods: List[int] = Field(default_factory=lambda: [5, 10, 20, 50, 200])
    ema_periods: List[int] = Field(default_factory=lambda: [5, 10, 20, 50, 200])

    @model_validator(mode="before")
    @classmethod
    def parse_price_features(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, list):
            enabled = {item for item in value if isinstance(item, str)}
            return {"enabled": enabled}
        if isinstance(value, dict):
            enabled = {
                name
                for name, flag in value.items()
                if name not in {"sma_periods", "ema_periods"} and flag
            }
            parsed = {"enabled": enabled}
            if "sma_periods" in value:
                parsed["sma_periods"] = value["sma_periods"]
            if "ema_periods" in value:
                parsed["ema_periods"] = value["ema_periods"]
            return parsed
        raise ValueError(
            "price_features must be a list of feature names or mapping of booleans"
        )


class VolumeMovingAverageConfig(BaseModel):
    periods: List[int] = Field(default_factory=list)


class VolumeFeaturesConfig(BaseModel):
    volume_sma: VolumeMovingAverageConfig = Field(
        default_factory=VolumeMovingAverageConfig
    )
    volume_ema: VolumeMovingAverageConfig = Field(
        default_factory=VolumeMovingAverageConfig
    )
    volume_sma_ratios: List[int] = Field(default_factory=list)
    volume_ema_ratios: List[int] = Field(default_factory=list)
    on_balance_volume: bool = True
    volume_price_trend: bool = False

    @model_validator(mode="before")
    @classmethod
    def parse_volume_features(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        parsed: Dict[str, Any] = {}
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str):
                if item == "on_balance_volume":
                    parsed["on_balance_volume"] = True
                if item == "volume_price_trend":
                    parsed["volume_price_trend"] = True
            elif isinstance(item, dict):
                for key, entry in item.items():
                    parsed[key] = entry
        return parsed


class BollingerBandsConfig(BaseModel):
    period: int = 20
    std_dev: float = 2.0


class KeltnerChannelsConfig(BaseModel):
    periods: List[int] = Field(default_factory=list)
    atr_multiple: float = 2.0


class VolatilityFeaturesConfig(BaseModel):
    atr_periods: List[int] = Field(default_factory=list)
    bollinger_bands: Optional[BollingerBandsConfig] = Field(
        default_factory=BollingerBandsConfig
    )
    keltner_channels: Optional[KeltnerChannelsConfig] = Field(
        default_factory=KeltnerChannelsConfig
    )

    @model_validator(mode="before")
    @classmethod
    def parse_volatility_features(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        parsed: Dict[str, Any] = {}
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict):
                continue
            for key, entry in item.items():
                parsed[key] = entry
        if isinstance(value, dict):
            parsed.update({k: v for k, v in value.items() if k not in parsed})
        return parsed


class VolatilityBreakoutConfig(BaseModel):
    lookback: List[int] = Field(default_factory=list)
    threshold: float = 2.0


class CustomFeaturesConfig(BaseModel):
    price_momentum: List[int] = Field(default_factory=list)
    volume_momentum: List[int] = Field(default_factory=list)
    mean_reversion: List[int] = Field(default_factory=list)
    volatility_breakout: Optional[VolatilityBreakoutConfig] = None

    @model_validator(mode="before")
    @classmethod
    def parse_custom_features(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        parsed: Dict[str, Any] = {}
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict):
                for key, entry in item.items():
                    if isinstance(entry, dict) and key != "volatility_breakout":
                        if "periods" in entry:
                            parsed[key] = entry.get("periods", [])
                        elif "lookback" in entry:
                            parsed[key] = entry.get("lookback", [])
                        else:
                            parsed[key] = entry
                    else:
                        parsed[key] = entry
        return parsed


class FeaturesConfigSchema(BaseModel):
    pipeline: PipelineConfig
    price_features: PriceFeaturesConfig = Field(default_factory=PriceFeaturesConfig)
    volume_features: VolumeFeaturesConfig = Field(default_factory=VolumeFeaturesConfig)
    volatility_features: VolatilityFeaturesConfig = Field(
        default_factory=VolatilityFeaturesConfig
    )
    custom_features: CustomFeaturesConfig = Field(default_factory=CustomFeaturesConfig)
    feature_combinations: Optional[Any] = None
    sentiment: Optional[SentimentConfig] = None
    feature_selection: Optional[Dict[str, Any]] = None
    preprocessing: Optional[Dict[str, Any]] = None
    multi_timeframe_features: Optional[MultiTimeframeConfig] = None


class TransactionCostConfig(BaseModel):
    enabled: bool = False
    mode: Literal["bps", "fixed"] = "bps"
    value: float = 0.0
    apply_on: Literal["notional", "shares"] = "notional"


class SlippageConfig(BaseModel):
    enabled: bool = False
    mode: Literal["bps", "fixed"] = "bps"
    value: float = 0.0
    reference_price: Literal["close", "mid"] = "close"


class LiquidityConfig(BaseModel):
    enabled: bool = False
    max_participation: float = 0.1
    volume_source: str = "bar_volume"
    order_book_depth: float | None = None


class MarketImpactConfig(BaseModel):
    enabled: bool = False
    model: Literal["linear", "square_root", "almgren_chriss"] = "linear"
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float | None = None
    decay: float = 0.0
    decay_volume_coeff: float = 0.0
    spread: float = 0.0
    spread_model: Optional[Dict[str, Any]] = None
    alpha_buy: float | None = None
    alpha_sell: float | None = None
    beta_buy: float | None = None
    beta_sell: float | None = None
    cross_alpha: float = 0.0
    cross_beta: float = 0.0
    horizon_decay: float = 0.0
    average_daily_volume: float | None = None
    liquidity_scale: float = Field(1.0, ge=0.0)


class BorrowFeeConfig(BaseModel):
    enabled: bool = False
    rate_bps: float = Field(0.0, ge=0.0)


class IntrabarConfig(BaseModel):
    enabled: bool = False
    tick_column: str = "ticks"
    drift: float = 0.0
    volatility: float = 0.0
    synthetic_ticks: int = 0


class ExecutionConfig(BaseModel):
    transaction_costs: TransactionCostConfig = TransactionCostConfig()
    slippage: SlippageConfig = SlippageConfig()
    liquidity: LiquidityConfig = LiquidityConfig()
    impact: MarketImpactConfig = MarketImpactConfig()
    borrow_fee: BorrowFeeConfig = BorrowFeeConfig()
    intrabar: IntrabarConfig = IntrabarConfig()


class DrawdownProtectionConfig(BaseModel):
    enabled: bool = False
    max_drawdown_pct: float | None = None
    max_drawdown_absolute: float | None = None
    warning_threshold: float = 0.8
    soft_stop_threshold: float = 0.9
    hard_stop_threshold: float = 1.0
    emergency_stop_threshold: float = 1.1
    lookback_periods: List[int] = [1, 7, 30]


class TurnoverLimitsConfig(BaseModel):
    daily_max: float | None = None
    weekly_max: float | None = None
    monthly_max: float | None = None


class RiskManagementConfig(BaseModel):
    drawdown_protection: DrawdownProtectionConfig = DrawdownProtectionConfig()
    turnover_limits: TurnoverLimitsConfig = TurnoverLimitsConfig()


class BacktestConfigSchema(BaseModel):
    data_path: str
    execution: ExecutionConfig = ExecutionConfig()


class PositionManagerConfig(BaseModel):
    """Configuration for real-time position management."""

    risk_management: RiskManagementConfig = RiskManagementConfig()
    impact: MarketImpactConfig = MarketImpactConfig()
    reconciliation: Dict[str, str] = {"intraday": "1m", "daily": "1d"}
    mode: Literal["paper", "live"] = "paper"
