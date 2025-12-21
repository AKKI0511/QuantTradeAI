"""End-to-end live trading orchestrator.

This module stitches together streaming ingestion, feature generation,
model inference, risk controls, and execution hooks to deliver a
production-ready asynchronous pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import yaml

from quanttradeai.data.processor import DataProcessor
from quanttradeai.models.classifier import MomentumClassifier
from quanttradeai.streaming.gateway import StreamingGateway
from quanttradeai.streaming.logging import logger
from quanttradeai.trading.drawdown_guard import DrawdownGuard
from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.trading.risk_manager import RiskManager
from quanttradeai.utils.config_schemas import (
    PositionManagerConfig,
    RiskManagementConfig,
)
from quanttradeai.trading.position_manager import PositionManager

ExecutionHook = Callable[[dict], None]


def _load_risk_guard(risk_config: str | None) -> DrawdownGuard | None:
    if not risk_config:
        return None
    path = Path(risk_config)
    if not path.exists():
        raise FileNotFoundError(f"Risk config not found: {risk_config}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = RiskManagementConfig(**raw.get("risk_management", raw))
    if cfg.drawdown_protection.enabled:
        return DrawdownGuard(cfg)
    return None


def _load_position_manager(
    config_path: str | None, cash: float
) -> PositionManager | None:
    if config_path is None:
        return None
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Position manager config not found: {config_path}")
    cfg_raw = yaml.safe_load(cfg_path.read_text()) or {}
    cfg = cfg_raw.get("position_manager", cfg_raw)
    pm_config = PositionManagerConfig(**cfg)
    pm = PositionManager.from_config(pm_config)
    pm.cash = cash
    return pm


def _load_streaming_symbols(config_path: str) -> list[str]:
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    streaming_cfg = cfg.get("streaming", {})
    symbols = streaming_cfg.get("symbols", []) or []
    for provider in streaming_cfg.get("providers", []) or []:
        symbols.extend(provider.get("symbols", []) or [])
    return sorted(set(symbols))


@dataclass
class LiveTradingEngine:
    """Coordinate streaming ingestion, inference, risk, and execution."""

    model_config: str
    model_path: str
    streaming_config: str = "config/streaming.yaml"
    risk_config: str | None = "config/risk_config.yaml"
    position_manager_config: str | None = "config/position_manager.yaml"
    enable_health_api: bool | None = None
    initial_capital: float = 1_000_000.0
    history_window: int = 512
    min_history_for_features: int = 220
    stop_loss_pct: float = 0.01
    execution_hook: ExecutionHook | None = None
    gateway: StreamingGateway | None = None
    data_processor: DataProcessor | None = None
    model: MomentumClassifier | None = None
    _history: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    _consumer: asyncio.Task | None = field(default=None, init=False)
    execution_log: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.gateway = self.gateway or StreamingGateway(self.streaming_config)
        if self.enable_health_api is not None:
            self.gateway._api_enabled = self.enable_health_api  # type: ignore[attr-defined]
        self.data_processor = self.data_processor or DataProcessor()
        self.model = self.model or MomentumClassifier(self.model_config)
        if not self.model_path:
            raise ValueError("model_path is required for live trading.")
        self.model.load_model(self.model_path)
        guard = _load_risk_guard(self.risk_config)
        self.risk_manager = RiskManager(drawdown_guard=guard)
        self.portfolio = PortfolioManager(
            capital=self.initial_capital, risk_manager=self.risk_manager
        )
        self.position_manager = _load_position_manager(
            self.position_manager_config, cash=self.initial_capital
        )
        if self.position_manager:
            try:
                symbols = _load_streaming_symbols(self.streaming_config)
                if symbols:
                    self.position_manager.bind_gateway(self.gateway, symbols)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("failed_binding_position_manager", error=str(exc))

    @property
    def health_monitor(self):
        return self.gateway.health_monitor

    def _extract_timestamp(self, message: dict) -> pd.Timestamp:
        ts = (
            message.get("timestamp")
            or message.get("time")
            or message.get("Datetime")
            or message.get("Date")
        )
        if ts is None:
            return pd.Timestamp(datetime.now(UTC))
        return (
            pd.to_datetime(ts).tz_localize(UTC)
            if not pd.Timestamp(ts).tzinfo
            else pd.Timestamp(ts)
        )

    def _extract_price_fields(
        self, message: dict
    ) -> tuple[float, float, float, float, float]:
        price = message.get("price") or message.get("close") or message.get("Close")
        open_price = message.get("open") or message.get("Open") or price
        high = message.get("high") or message.get("High") or price
        low = message.get("low") or message.get("Low") or price
        close = message.get("close") or message.get("Close") or price
        volume = message.get("volume") or message.get("Volume") or 0.0
        return float(open_price), float(high), float(low), float(close), float(volume)

    def _update_history(self, message: dict) -> None:
        symbol = message.get("symbol")
        if not symbol:
            return
        ts = self._extract_timestamp(message)
        open_p, high, low, close, vol = self._extract_price_fields(message)
        row = pd.DataFrame(
            [{"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": vol}],
            index=[ts],
        )
        history = self._history.get(symbol)
        history = pd.concat([history, row]) if history is not None else row
        self._history[symbol] = history.tail(self.history_window)

    def _prepare_features(self, symbol: str) -> Optional[pd.DataFrame]:
        history = self._history.get(symbol)
        if history is None or len(history) < self.min_history_for_features:
            return None
        processed = self.data_processor.process_data(history.copy())
        if processed.empty:
            return None
        return processed.tail(1)

    def _predict_signal(self, features: pd.DataFrame) -> Optional[int]:
        if self.model is None or self.model.model is None:
            raise ValueError("Model is not loaded for inference.")
        feature_cols = self.model.feature_columns
        if not feature_cols:
            feature_cols = [
                c
                for c in features.columns
                if c not in {"Open", "High", "Low", "Close", "Volume"}
            ]
        missing = [c for c in feature_cols if c not in features.columns]
        if missing:
            logger.warning("missing_feature_columns", columns=missing)
            return None
        X = features[feature_cols].values
        prediction = self.model.predict(X)
        return int(prediction[0])

    def _sync_position_manager(self) -> None:
        if self.position_manager is not None:
            self.position_manager.cash = self.portfolio.cash

    def _record_execution(self, payload: dict) -> None:
        self.execution_log.append(payload)
        if self.execution_hook is not None:
            try:
                self.execution_hook(payload)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("execution_hook_failed", error=str(exc))

    def _handle_signal(self, symbol: str, price: float, signal: int) -> None:
        if signal > 0:
            if symbol in self.portfolio.positions:
                return
            qty = self.portfolio.open_position(
                symbol, price, stop_loss_pct=self.stop_loss_pct
            )
            self._sync_position_manager()
            if qty > 0:
                if self.position_manager is not None:
                    self.position_manager.open_position(symbol, qty, price)
                self._record_execution(
                    {"action": "buy", "symbol": symbol, "qty": qty, "price": price}
                )
        elif signal < 0 and symbol in self.portfolio.positions:
            qty = self.portfolio.close_position(symbol, price)
            self._sync_position_manager()
            if qty > 0:
                if self.position_manager is not None:
                    self.position_manager.close_position(symbol, price)
                self._record_execution(
                    {"action": "sell", "symbol": symbol, "qty": qty, "price": price}
                )

    async def _consume_buffer(self) -> None:
        while True:
            message = await self.gateway.buffer.get()
            start = time.perf_counter()
            try:
                symbol = message.get("symbol")
                if not symbol:
                    continue
                self._update_history(message)
                price = float(
                    message.get("price")
                    or message.get("close")
                    or message.get("Close")
                    or message.get("Mid")
                    or 0.0
                )
                features = self._prepare_features(symbol)
                if features is None:
                    continue
                signal = self._predict_signal(features)
                if signal is None:
                    continue
                self._handle_signal(symbol, price, signal)
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.error("live_pipeline_error", error=str(exc))
                self.health_monitor.trigger_alerts(
                    "error", f"live_pipeline_error: {exc}"
                )
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                try:
                    self.health_monitor.metrics_collector.record_latency(
                        "inference", elapsed_ms
                    )
                    depth = self.gateway.buffer.queue.qsize()
                    self.health_monitor.metrics_collector.record_queue_depth(
                        "stream", depth
                    )
                except Exception:  # pragma: no cover - metrics best effort
                    pass

    async def start(self) -> None:
        """Start streaming and inference concurrently."""

        self._consumer = asyncio.create_task(self._consume_buffer())
        try:
            await self.gateway._start()
        finally:
            if self._consumer:
                self._consumer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._consumer
