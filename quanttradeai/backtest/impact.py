"""Market impact modeling utilities.

Provides a base :class:`MarketImpactModel` and a set of concrete
implementations used to estimate execution price effects. These models are
kept deliberately lightweight to allow fast vectorised backtests.

The module also exposes :class:`ImpactCalculator` which orchestrates model
selection and applies spread and decay logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
import math


class MarketImpactModel(Protocol):
    """Interface for market impact models."""

    alpha: float
    beta: float

    def temporary(self, trade_volume: float, adv: float) -> float:
        """Return temporary impact per share."""

    def permanent(self, trade_volume: float, adv: float) -> float:
        """Return permanent impact per share."""


@dataclass
class LinearImpactModel:
    """Simple square-root impact model."""

    alpha: float = 0.0
    beta: float = 0.0

    def _ratio(self, trade_volume: float, adv: float) -> float:
        if adv <= 0:
            return 0.0
        return trade_volume / adv

    def temporary(
        self, trade_volume: float, adv: float
    ) -> float:  # pragma: no cover - simple
        r = self._ratio(trade_volume, adv)
        return self.alpha * math.sqrt(r)

    def permanent(self, trade_volume: float, adv: float) -> float:
        r = self._ratio(trade_volume, adv)
        return self.beta * r


@dataclass
class SquareRootImpactModel(LinearImpactModel):
    """Square-root model with explicit temporary/permanent components."""

    def temporary(self, trade_volume: float, adv: float) -> float:
        r = self._ratio(trade_volume, adv)
        return self.alpha * math.sqrt(r)

    def permanent(self, trade_volume: float, adv: float) -> float:
        r = self._ratio(trade_volume, adv)
        return self.beta * math.sqrt(r)


@dataclass
class AlmgrenChrissModel(LinearImpactModel):
    """Basic Almgren-Chriss style model."""

    gamma: float = 0.0

    def temporary(self, trade_volume: float, adv: float) -> float:
        r = self._ratio(trade_volume, adv)
        return self.alpha * r + self.beta * math.sqrt(r)

    def permanent(self, trade_volume: float, adv: float) -> float:
        r = self._ratio(trade_volume, adv)
        return self.gamma * r


MODEL_MAP = {
    "linear": LinearImpactModel,
    "square_root": SquareRootImpactModel,
    "almgren_chriss": AlmgrenChrissModel,
}


@dataclass
class DynamicSpreadModel:
    """Adaptive bid-ask spread model."""

    def __init__(
        self,
        base: float = 0.0,
        vol_coeff: float = 0.0,
        volume_coeff: float = 0.0,
        tod: dict[int, float] | None = None,
    ) -> None:
        self.base = base
        self.vol_coeff = vol_coeff
        self.volume_coeff = volume_coeff
        self.tod = tod or {}

    def value(self, volatility: float, volume: float, ts: datetime) -> float:
        spread = self.base + self.vol_coeff * volatility
        if volume > 0:
            spread += self.volume_coeff / volume
        if self.tod:
            spread *= self.tod.get(ts.hour, 1.0)
        return spread


@dataclass
class ImpactCalculator:
    """Service class for applying market impact models."""

    model: MarketImpactModel
    decay: float = 0.0
    decay_volume_coeff: float = 0.0
    spread: float = 0.0
    spread_model: DynamicSpreadModel | None = None
    alpha_buy: float | None = None
    alpha_sell: float | None = None
    beta_buy: float | None = None
    beta_sell: float | None = None
    cross_alpha: float = 0.0
    cross_beta: float = 0.0
    horizon_decay: float = 0.0
    _cache: dict[tuple[float, float, int], tuple[float, float]] = field(
        default_factory=dict, init=False
    )
    _perm_state: float = field(default=0.0, init=False)

    def _impact(
        self, trade_volume: float, adv: float, side: int, cross_volume: float
    ) -> tuple[float, float]:
        key = (trade_volume, adv, side)
        if key in self._cache:
            tmp, perm = self._cache[key]
        else:
            # apply asymmetry
            orig_alpha, orig_beta = self.model.alpha, self.model.beta
            if side > 0:
                if self.alpha_buy is not None:
                    self.model.alpha = self.alpha_buy
                if self.beta_buy is not None:
                    self.model.beta = self.beta_buy
            else:
                if self.alpha_sell is not None:
                    self.model.alpha = self.alpha_sell
                if self.beta_sell is not None:
                    self.model.beta = self.beta_sell
            tmp = self.model.temporary(trade_volume, adv)
            perm = self.model.permanent(trade_volume, adv)
            self.model.alpha, self.model.beta = orig_alpha, orig_beta
            self._cache[key] = (tmp, perm)
        if self.decay > 0 or self.decay_volume_coeff > 0:
            decay = self.decay
            if self.decay_volume_coeff > 0:
                decay *= trade_volume**self.decay_volume_coeff
            tmp *= math.exp(-decay)
        if self.cross_alpha or self.cross_beta:
            r_c = cross_volume / adv if adv > 0 else 0.0
            tmp += self.cross_alpha * math.sqrt(r_c)
            perm += self.cross_beta * math.sqrt(r_c)
        if self.horizon_decay:
            perm += self._perm_state
            self._perm_state = self._perm_state * self.horizon_decay + perm
        return tmp, perm

    def impact_per_share(
        self,
        trade_volume: float,
        adv: float,
        *,
        side: int = 1,
        volatility: float = 0.0,
        volume: float = 0.0,
        timestamp: datetime | None = None,
        cross_volume: float = 0.0,
    ) -> tuple[float, float, float]:
        tmp, perm = self._impact(trade_volume, adv, side, cross_volume)
        if self.spread_model is not None and timestamp is not None:
            spread_val = self.spread_model.value(volatility, volume, timestamp)
        else:
            spread_val = self.spread
        spread_cost = spread_val / 2.0
        return tmp, perm, spread_cost

    def impact_cost(
        self,
        trade_volume: float,
        adv: float,
        *,
        side: int = 1,
        volatility: float = 0.0,
        volume: float = 0.0,
        timestamp: datetime | None = None,
        cross_volume: float = 0.0,
    ) -> dict[str, float]:
        tmp, perm, spread_cost = self.impact_per_share(
            trade_volume,
            adv,
            side=side,
            volatility=volatility,
            volume=volume,
            timestamp=timestamp,
            cross_volume=cross_volume,
        )
        per_share = tmp + spread_cost + perm
        return {
            "temp": tmp,
            "perm": perm,
            "spread": spread_cost,
            "total": per_share * trade_volume,
        }
