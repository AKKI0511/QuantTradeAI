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


class ProjectStreamingMonitorConfig(BaseModel):
    enabled: bool = True
    check_interval: int = Field(5, ge=1)
    metrics_retention: Optional[int] = Field(default=None, ge=1)


class ProjectStreamingThresholdsConfig(BaseModel):
    max_latency_ms: Optional[float] = Field(default=None, ge=0.0)
    min_throughput_msg_per_sec: Optional[float] = Field(default=None, ge=0.0)
    max_reconnect_attempts: Optional[int] = Field(default=None, ge=1)
    max_queue_depth: Optional[int] = Field(default=None, ge=1)
    circuit_breaker_timeout: Optional[int] = Field(default=None, ge=1)


class ProjectStreamingAlertsConfig(BaseModel):
    enabled: bool = True
    channels: List[str] = Field(default_factory=lambda: ["log"])
    escalation_threshold: int = Field(3, ge=1)


class ProjectStreamingMetricsConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = Field(9000, ge=1, le=65535)


class ProjectStreamingAPIConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)


class ProjectDataStreamingConfig(BaseModel):
    enabled: bool = False
    provider: Optional[str] = None
    websocket_url: Optional[str] = None
    auth_method: str = "api_key"
    symbols: List[str] = Field(default_factory=list)
    channels: List[Literal["trades", "quotes"]] = Field(default_factory=list)
    buffer_size: int = Field(1000, ge=1)
    reconnect_attempts: int = Field(5, ge=1)
    health_check_interval: Optional[int] = Field(default=30, ge=1)
    rate_limit: Dict[str, Any] = Field(default_factory=dict)
    circuit_breaker: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Optional[ProjectStreamingMonitorConfig] = None
    thresholds: Optional[ProjectStreamingThresholdsConfig] = None
    alerts: Optional[ProjectStreamingAlertsConfig] = None
    metrics: Optional[ProjectStreamingMetricsConfig] = None
    api: Optional[ProjectStreamingAPIConfig] = None

    @model_validator(mode="after")
    def validate_enabled_requirements(self) -> "ProjectDataStreamingConfig":
        if not self.enabled:
            return self
        missing = []
        if not self.provider:
            missing.append("provider")
        if not self.websocket_url:
            missing.append("websocket_url")
        if not self.symbols:
            missing.append("symbols")
        if not self.channels:
            missing.append("channels")
        if missing:
            raise ValueError(
                "data.streaming requires "
                + ", ".join(missing)
                + " when streaming is enabled."
            )
        return self


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
    streaming: Optional[ProjectDataStreamingConfig] = None

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
    bollinger_bands: Optional[BollingerBandsConfig] = None
    keltner_channels: Optional[KeltnerChannelsConfig] = None

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


class ProjectSection(BaseModel):
    name: str
    profile: str


class ProjectProfileSection(BaseModel):
    mode: str


class FeatureDefinition(BaseModel):
    name: str
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ProjectFeaturesSection(BaseModel):
    definitions: List[FeatureDefinition] = Field(default_factory=list)


class ProjectResearchLabelsConfig(BaseModel):
    type: Literal["forward_return"] = "forward_return"
    horizon: int = 5
    buy_threshold: float = 0.01
    sell_threshold: float = -0.01


class ProjectResearchTuningConfig(BaseModel):
    enabled: bool = True
    trials: int = Field(50, ge=1)


class ProjectResearchModelConfig(BaseModel):
    kind: Literal["classifier"] = "classifier"
    family: Literal["voting"] = "voting"
    tuning: ProjectResearchTuningConfig = Field(
        default_factory=ProjectResearchTuningConfig
    )


class ProjectResearchEvaluationConfig(BaseModel):
    split: Literal["time_aware"] = "time_aware"
    use_configured_test_window: bool = True


class ProjectResearchBacktestCostsConfig(BaseModel):
    enabled: bool = False
    bps: float = Field(0.0, ge=0.0)


class ProjectResearchBacktestConfig(BaseModel):
    costs: ProjectResearchBacktestCostsConfig = Field(
        default_factory=ProjectResearchBacktestCostsConfig
    )


class ProjectResearchSection(BaseModel):
    enabled: bool = True
    labels: ProjectResearchLabelsConfig = Field(
        default_factory=ProjectResearchLabelsConfig
    )
    model: ProjectResearchModelConfig = Field(
        default_factory=ProjectResearchModelConfig
    )
    evaluation: ProjectResearchEvaluationConfig = Field(
        default_factory=ProjectResearchEvaluationConfig
    )
    backtest: ProjectResearchBacktestConfig = Field(
        default_factory=ProjectResearchBacktestConfig
    )


class ProjectAgentLLMConfig(BaseModel):
    provider: str
    model: str
    prompt_file: str
    api_key_env_var: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class ProjectAgentModelConfig(BaseModel):
    path: str


class ProjectAgentMarketDataContext(BaseModel):
    enabled: bool = True
    timeframe: str = "1d"
    lookback_bars: int = Field(20, gt=0)

    @model_validator(mode="before")
    @classmethod
    def parse_bool(cls, value: Any) -> Dict[str, Any]:
        if value in (None, False):
            return {"enabled": False}
        if value is True:
            return {"enabled": True}
        return value


class ProjectModelSignalSourceConfig(BaseModel):
    name: str
    path: str


class ProjectAgentContextConfig(BaseModel):
    market_data: Optional[ProjectAgentMarketDataContext] = None
    features: List[str] = Field(default_factory=list)
    model_signals: List[str] = Field(default_factory=list)
    positions: bool = False
    orders: bool = False
    risk_state: bool = False
    news: bool = False
    memory: bool = False
    notes: bool = False


class ProjectAgentConfig(BaseModel):
    name: str
    kind: Literal["rule", "model", "llm", "hybrid"]
    mode: Literal["backtest", "paper", "live"]
    llm: Optional[ProjectAgentLLMConfig] = None
    model: Optional[ProjectAgentModelConfig] = None
    context: ProjectAgentContextConfig = Field(
        default_factory=ProjectAgentContextConfig
    )
    tools: List[Literal["get_quote", "get_position", "place_order"]] = Field(
        default_factory=list
    )
    risk: Dict[str, Any] = Field(default_factory=dict)
    model_signal_sources: List[ProjectModelSignalSourceConfig | str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_requirements(self) -> "ProjectAgentConfig":
        if self.kind in {"llm", "hybrid"} and self.llm is None:
            raise ValueError(
                f"agents.{self.name} requires an llm block when kind is {self.kind}."
            )
        if self.kind == "model" and self.model is None:
            raise ValueError(
                f"agents.{self.name} requires a model block when kind is model."
            )
        if self.kind != "model" and self.model is not None:
            raise ValueError(
                f"agents.{self.name} only supports a model block when kind is model."
            )
        if self.kind not in {"llm", "hybrid"} and self.llm is not None:
            raise ValueError(
                f"agents.{self.name} only supports an llm block when kind is llm or hybrid."
            )
        return self


class ProjectDeploymentSection(BaseModel):
    target: str
    mode: str


class ProjectConfigSchema(BaseModel):
    project: ProjectSection
    profiles: Dict[str, ProjectProfileSection]
    data: DataSection
    features: ProjectFeaturesSection
    research: ProjectResearchSection
    agents: List[ProjectAgentConfig]
    deployment: ProjectDeploymentSection

    @model_validator(mode="after")
    def validate_paper_streaming_requirements(self) -> "ProjectConfigSchema":
        requires_streaming = any(
            agent.kind == "model" and agent.mode == "paper" for agent in self.agents
        )
        if requires_streaming and (
            self.data.streaming is None or not self.data.streaming.enabled
        ):
            raise ValueError(
                "data.streaming.enabled must be true when a model agent is configured with mode=paper."
            )
        return self
