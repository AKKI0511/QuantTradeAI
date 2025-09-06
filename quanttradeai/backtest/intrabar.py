"""Intrabar price path simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np


@dataclass
class BrownianParams:
    """Parameters for geometric Brownian motion tick generation."""

    drift: float = 0.0
    volatility: float = 0.0
    ticks: int = 0


def generate_gbm_ticks(
    start_price: float,
    volume: float,
    params: BrownianParams,
    seed: int | None = None,
) -> Sequence[dict[str, float]]:
    """Generate synthetic ticks using geometric Brownian motion.

    Parameters
    ----------
    start_price : float
        Starting price for the path.
    volume : float
        Total volume available in the bar.
    params : BrownianParams
        Brownian motion parameters controlling drift, volatility and number of
        ticks to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Sequence[dict[str, float]]
        List of tick dictionaries with ``price`` and ``volume`` keys.
    """
    n = max(1, params.ticks)
    dt = 1.0 / n
    rng = np.random.default_rng(seed)
    prices = [start_price]
    for _ in range(n - 1):
        shock = (
            params.drift - 0.5 * params.volatility**2
        ) * dt + params.volatility * math.sqrt(dt) * rng.standard_normal()
        prices.append(prices[-1] * math.exp(shock))
    vol_per_tick = volume / n if n else 0.0
    return [{"price": p, "volume": vol_per_tick} for p in prices]
