from __future__ import annotations

"""Alert management utilities for streaming health monitoring."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple
import logging
import time

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
    escalation_threshold: int = 3
    callbacks: List[Callable[[str, str], None]] = field(default_factory=list)
    warning_counts: Dict[str, int] = field(default_factory=dict)
    history: List[Tuple[float, str, str]] = field(default_factory=list)

    def _dispatch(self, level: str, message: str) -> None:
        if "log" in self.channels:
            log_fn = getattr(logger, level, logger.warning)
            log_fn(message)
        if "metrics" in self.channels:
            # Integration point for metric-based alerting (e.g. Prometheus
            # counters).  Left as a no-op for lightweight deployments.
            pass
        for cb in self.callbacks:
            cb(level, message)

    def send(self, level: str, message: str) -> None:
        """Send an alert with ``level`` and ``message``.

        Escalates after ``escalation_threshold`` warnings and records a simple
        in-memory incident history for later inspection.
        """

        self.history.append((time.time(), level, message))
        if level == "warning":
            count = self.warning_counts.get(message, 0) + 1
            self.warning_counts[message] = count
            if count >= self.escalation_threshold:
                self.warning_counts[message] = 0
                self._dispatch("error", message)
                return
        self._dispatch(level, message)
