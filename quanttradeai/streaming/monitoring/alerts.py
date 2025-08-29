from __future__ import annotations

"""Alert management utilities for streaming health monitoring."""

from dataclasses import dataclass, field
from typing import Callable, Iterable, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlertManager:
    """Dispatch alerts through configurable channels.

    Parameters
    ----------
    channels:
        Iterable of channels to emit alerts on. Supported values are
        ``"log"`` and ``"metrics"``.  Additional channels can be plugged in by
        registering callback functions via :attr:`callbacks`.
    """

    channels: Iterable[str] = ("log",)
    callbacks: List[Callable[[str, str], None]] = field(default_factory=list)

    def send(self, level: str, message: str) -> None:
        """Send an alert with ``level`` and ``message``.

        The method logs the alert and notifies any registered callbacks.  The
        ``metrics`` channel is a placeholder for integration with external
        monitoring systems and currently acts as a no-op.
        """

        if "log" in self.channels:
            log_fn = getattr(logger, level, logger.warning)
            log_fn(message)
        if "metrics" in self.channels:
            # Integration point for metric-based alerting (e.g. Prometheus
            # counters).  Left as a no-op for lightweight deployments.
            pass
        for cb in self.callbacks:
            cb(level, message)
