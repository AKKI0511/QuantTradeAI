"""Market impact modeling utilities.

Provides a base :class:`MarketImpactModel` and a set of concrete
implementations used to estimate execution price effects. These models are
kept deliberately lightweight to allow fast vectorised backtests.

The module also exposes :class:`ImpactCalculator` which orchestrates model
selection and applies spread and decay logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class ImpactCalculator:
    """Service class for applying market impact models."""

    model: MarketImpactModel
    decay: float = 0.0
    spread: float = 0.0
    _cache: dict[tuple[float, float], tuple[float, float]] = field(
        default_factory=dict, init=False
    )

    def _impact(self, trade_volume: float, adv: float) -> tuple[float, float]:
        key = (trade_volume, adv)
        if key in self._cache:
            return self._cache[key]
        tmp = self.model.temporary(trade_volume, adv)
        perm = self.model.permanent(trade_volume, adv)
        if self.decay > 0:
            tmp *= math.exp(-self.decay)
        self._cache[key] = (tmp, perm)
        return tmp, perm

    def impact_per_share(
        self, trade_volume: float, adv: float
    ) -> tuple[float, float, float]:
        tmp, perm = self._impact(trade_volume, adv)
        spread_cost = self.spread / 2.0
        return tmp, perm, spread_cost

    def impact_cost(self, trade_volume: float, adv: float) -> dict[str, float]:
        tmp, perm, spread_cost = self.impact_per_share(trade_volume, adv)
        per_share = tmp + spread_cost + perm
        return {
            "temp": tmp,
            "perm": perm,
            "spread": spread_cost,
            "total": per_share * trade_volume,
        }
