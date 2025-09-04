"""Pydantic configuration schemas.

Defines validation models for data and feature configuration files used
throughout the project.

Key Components:
    - :class:`DataSection`
    - :class:`ModelConfigSchema`
    - :class:`FeaturesConfigSchema`
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel


class DataSection(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    timeframe: Optional[str] = "1d"
    cache_path: Optional[str] = None
    cache_dir: Optional[str] = None
    cache_expiration_days: Optional[int] = None
    use_cache: Optional[bool] = True
    refresh: Optional[bool] = False
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    max_workers: Optional[int] = 1


class ModelConfigSchema(BaseModel):
    data: DataSection


class PipelineConfig(BaseModel):
    steps: List[str]


class SentimentConfig(BaseModel):
    """Configuration for LLM-based sentiment analysis."""

    enabled: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key_env_var: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class FeaturesConfigSchema(BaseModel):
    pipeline: PipelineConfig
    price_features: Optional[Any] = None
    volume_features: Optional[Any] = None
    volatility_features: Optional[Any] = None
    custom_features: Optional[Any] = None
    feature_combinations: Optional[Any] = None
    sentiment: Optional[SentimentConfig] = None
    feature_selection: Optional[Dict[str, Any]] = None
    preprocessing: Optional[Dict[str, Any]] = None


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


class MarketImpactConfig(BaseModel):
    enabled: bool = False
    model: Literal["linear", "square_root", "almgren_chriss"] = "linear"
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float | None = None
    decay: float = 0.0
    spread: float = 0.0
    average_daily_volume: float | None = None


class ExecutionConfig(BaseModel):
    transaction_costs: TransactionCostConfig = TransactionCostConfig()
    slippage: SlippageConfig = SlippageConfig()
    liquidity: LiquidityConfig = LiquidityConfig()
    impact: MarketImpactConfig = MarketImpactConfig()


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
