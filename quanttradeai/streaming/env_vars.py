"""Environment-variable naming helpers for streaming providers."""

from __future__ import annotations

import re


def provider_env_var_prefix(provider: str) -> str:
    """Return normalized env var prefix for a provider name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(provider).upper()).strip("_")
