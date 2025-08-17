"""Basic message transformation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MessageProcessor:
    """Simple pass-through processor for streaming messages."""

    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Return the message unchanged.

        This placeholder allows future transformation logic.
        """

        return message
