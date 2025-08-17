"""Structured logging utilities for streaming infrastructure."""

from __future__ import annotations

import logging
import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for JSON formatted logs.

    Args:
        level: Minimum log level.
    """
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.contextvars.merge_contextvars,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
    )


configure_logging()
logger = structlog.get_logger()
