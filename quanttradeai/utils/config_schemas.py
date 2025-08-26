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


class ExecutionConfig(BaseModel):
    transaction_costs: TransactionCostConfig = TransactionCostConfig()
    slippage: SlippageConfig = SlippageConfig()
    liquidity: LiquidityConfig = LiquidityConfig()


class BacktestConfigSchema(BaseModel):
    data_path: str
    execution: ExecutionConfig = ExecutionConfig()
