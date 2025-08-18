"""Interactive Brokers data provider adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .base_adapter import DataProviderAdapter


@dataclass
class IBAdapter(DataProviderAdapter):
    """Adapter for Interactive Brokers TWS/Gateway streaming API."""

    name: str = "interactive_brokers"

    def _build_subscribe_message(
        self, channel: str, symbols: List[str]
    ) -> Dict[str, Any]:
        return {"action": "subscribe", "channel": channel, "symbols": symbols}
