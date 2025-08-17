"""Alpaca data provider adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .base_adapter import DataProviderAdapter


@dataclass
class AlpacaAdapter(DataProviderAdapter):
    """Adapter for Alpaca's streaming API."""

    name: str = "alpaca"

    def _build_subscribe_message(
        self, channel: str, symbols: List[str]
    ) -> Dict[str, Any]:
        return {"action": "subscribe", channel: symbols}
