"""Drawdown and turnover protection utilities.

Implements :class:`DrawdownGuard` which monitors portfolio value and trade
activity to enforce risk-based trading halts and position size reductions.

The guard supports configurable drawdown limits in absolute and percentage
terms, multi-level protection thresholds, and portfolio turnover caps. It is
thread-safe and suitable for both live trading and backtesting.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import yaml

from quanttradeai.utils.config_schemas import (
    DrawdownProtectionConfig,
    RiskManagementConfig,
    TurnoverLimitsConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class _RiskState:
    """Internal state used for thread-safe risk calculations."""

    drawdown_pct: float = 0.0
    drawdown_abs: float = 0.0
    size_multiplier: float = 1.0
    halt_trading: bool = False
    emergency: bool = False


class DrawdownGuard:
    """Monitor portfolio drawdowns and turnover limits."""

    def __init__(
        self,
        config: DrawdownProtectionConfig | RiskManagementConfig | Dict | None = None,
        turnover_limits: TurnoverLimitsConfig | None = None,
        config_path: str | None = None,
    ) -> None:
        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config = data.get("risk_management", data)

        if isinstance(config, RiskManagementConfig):
            dd_cfg = config.drawdown_protection
            to_cfg = config.turnover_limits
        elif isinstance(config, DrawdownProtectionConfig):
            dd_cfg = config
            to_cfg = turnover_limits or TurnoverLimitsConfig()
        elif isinstance(config, dict):
            dd_cfg = DrawdownProtectionConfig(
                **config.get("drawdown_protection", config)
            )
            to_cfg = TurnoverLimitsConfig(**config.get("turnover_limits", {}))
        else:
            dd_cfg = DrawdownProtectionConfig()
            to_cfg = turnover_limits or TurnoverLimitsConfig()

        self.config = dd_cfg
        self.turnover_limits = to_cfg

        self._lock = threading.Lock()
        self._high_water_mark: float = 0.0
        self._history: List[Tuple[datetime, float]] = []
        self._trades: List[Tuple[datetime, float]] = []
        self._state = _RiskState()

    def _rolling_drawdowns(self, now: datetime) -> Dict[int, float]:
        results: Dict[int, float] = {}
        for period in self.config.lookback_periods:
            cutoff = now - timedelta(days=period)
            values = [v for t, v in self._history if t >= cutoff]
            if not values:
                results[period] = 0.0
                continue
            peak = max(values)
            trough = min(values[values.index(peak) :]) if values else peak
            dd = (trough - peak) / peak if peak else 0.0
            results[period] = dd
        return results

    def _turnover_ratio(self, now: datetime, window: timedelta) -> float:
        cutoff = now - window
        total = sum(n for t, n in self._trades if t >= cutoff)
        limit = {
            timedelta(days=1): self.turnover_limits.daily_max,
            timedelta(days=7): self.turnover_limits.weekly_max,
            timedelta(days=30): self.turnover_limits.monthly_max,
        }[window]
        return total / limit if limit else 0.0

    def _evaluate_turnover(self, now: datetime) -> float:
        ratios = [
            self._turnover_ratio(now, timedelta(days=1)),
            self._turnover_ratio(now, timedelta(days=7)),
            self._turnover_ratio(now, timedelta(days=30)),
        ]
        ratios = [r for r in ratios if r > 0]
        return max(ratios) if ratios else 0.0

    def _update_state(self, ratio: float) -> None:
        cfg = self.config
        if cfg.emergency_stop_threshold and ratio >= cfg.emergency_stop_threshold:
            self._state.size_multiplier = 0.0
            self._state.halt_trading = True
            self._state.emergency = True
            logger.critical("Emergency stop triggered (ratio %.2f)", ratio)
        elif ratio >= cfg.hard_stop_threshold:
            self._state.size_multiplier = 0.0
            self._state.halt_trading = True
            self._state.emergency = False
            logger.critical("Hard stop triggered (ratio %.2f)", ratio)
        elif ratio >= cfg.soft_stop_threshold:
            self._state.size_multiplier = 0.5
            self._state.halt_trading = False
            self._state.emergency = False
            logger.warning("Soft stop triggered (ratio %.2f)", ratio)
        elif ratio >= cfg.warning_threshold:
            self._state.size_multiplier = 1.0
            logger.warning("Warning level reached (ratio %.2f)", ratio)
            self._state.halt_trading = False
            self._state.emergency = False
        else:
            self._state.size_multiplier = 1.0
            self._state.halt_trading = False
            self._state.emergency = False

    def update_portfolio_value(self, current_value: float, timestamp: datetime) -> None:
        with self._lock:
            if current_value <= 0:
                return
            self._history.append((timestamp, current_value))
            if current_value > self._high_water_mark:
                self._high_water_mark = current_value
            dd_abs = self._high_water_mark - current_value
            dd_pct = dd_abs / self._high_water_mark if self._high_water_mark else 0.0
            self._state.drawdown_abs = dd_abs
            self._state.drawdown_pct = dd_pct

            ratio = 0.0
            if self.config.max_drawdown_pct:
                ratio = max(ratio, dd_pct / self.config.max_drawdown_pct)
            if self.config.max_drawdown_absolute:
                ratio = max(ratio, dd_abs / self.config.max_drawdown_absolute)

            turnover_ratio = self._evaluate_turnover(timestamp)
            ratio = max(ratio, turnover_ratio)

            self._update_state(ratio)

    def record_trade(self, notional: float, timestamp: datetime) -> None:
        with self._lock:
            self._trades.append((timestamp, abs(notional)))
            ratio = self._evaluate_turnover(timestamp)
            dd_ratio = (
                self._state.drawdown_pct / self.config.max_drawdown_pct
                if self.config.max_drawdown_pct
                else 0.0
            )
            self._update_state(max(ratio, dd_ratio))

    def check_drawdown_limits(self) -> Dict[str, float | str]:
        with self._lock:
            now = datetime.utcnow()
            turnover_ratio = self._evaluate_turnover(now)
            dd_ratio = 0.0
            cfg = self.config
            if cfg.max_drawdown_pct:
                dd_ratio = max(
                    dd_ratio, self._state.drawdown_pct / cfg.max_drawdown_pct
                )
            if cfg.max_drawdown_absolute:
                dd_ratio = max(
                    dd_ratio, self._state.drawdown_abs / cfg.max_drawdown_absolute
                )
            ratio = max(dd_ratio, turnover_ratio)
            self._update_state(ratio)

            if self._state.emergency:
                status = "emergency_stop"
            elif self._state.halt_trading:
                status = "hard_stop"
            elif self._state.size_multiplier < 1.0:
                status = "soft_stop"
            elif ratio >= cfg.warning_threshold:
                status = "warning"
            else:
                status = "ok"

            return {
                "status": status,
                "ratio": ratio,
                "drawdown_pct": self._state.drawdown_pct,
                "drawdown_abs": self._state.drawdown_abs,
                "turnover_ratio": turnover_ratio,
            }

    def get_position_size_multiplier(self) -> float:
        with self._lock:
            return self._state.size_multiplier

    def should_halt_trading(self) -> bool:
        with self._lock:
            return self._state.halt_trading

    def should_emergency_liquidate(self) -> bool:
        with self._lock:
            return self._state.emergency

    def get_risk_metrics(self) -> Dict[str, float]:
        with self._lock:
            now = datetime.utcnow()
            metrics = {
                "high_water_mark": self._high_water_mark,
                "current_drawdown_pct": self._state.drawdown_pct,
                "current_drawdown_abs": self._state.drawdown_abs,
            }
            metrics.update(
                {f"mdd_{p}": v for p, v in self._rolling_drawdowns(now).items()}
            )
            return metrics

    def reset_high_water_mark(self) -> None:
        with self._lock:
            if self._history:
                _, last = self._history[-1]
                self._high_water_mark = last
            self._history.clear()
            self._trades.clear()
            self._state = _RiskState()
