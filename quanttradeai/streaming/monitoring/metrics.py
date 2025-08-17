"""Streaming metrics collection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    """Simple in-memory metrics tracker."""

    messages_received: int = 0

    def increment(self) -> None:
        """Increment the received message counter."""

        self.messages_received += 1
