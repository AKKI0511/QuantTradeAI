"""Validation utilities for streaming messages."""

from __future__ import annotations

from typing import Any, Dict


def validate_trade_message(message: Dict[str, Any]) -> bool:
    """Validate that a trade message contains required fields."""

    return "symbol" in message
