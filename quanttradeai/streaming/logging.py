"""Lightweight, structured logging for streaming.

This module prefers ``structlog`` for JSON logs, but gracefully falls back to
the standard library logger if ``structlog`` is unavailable. The exported
``logger`` exposes ``debug/info/warning/error`` so callers don’t care which
backend is in use.
"""

from __future__ import annotations

import logging

try:  # Prefer structlog when installed
    import structlog

    def configure_logging(level: str = "INFO") -> None:
        """Configure structlog for JSON formatted logs."""

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.contextvars.merge_contextvars,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level)
            ),
        )

    configure_logging()
    logger = structlog.get_logger()
except Exception:  # Fallback to stdlib logging without hard dependency
    _fallback = logging.getLogger("quanttradeai.streaming")
    _fallback.setLevel(logging.INFO)

    class _StdLogger:
        def debug(self, *args, **kwargs):
            _fallback.debug(*args, **kwargs)

        def info(self, *args, **kwargs):
            _fallback.info(*args, **kwargs)

        def warning(self, *args, **kwargs):
            _fallback.warning(*args, **kwargs)

        def error(self, *args, **kwargs):
            _fallback.error(*args, **kwargs)

    logger = _StdLogger()
