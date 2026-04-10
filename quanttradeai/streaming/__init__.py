"""Streaming infrastructure package.

Keep package imports lightweight so callers can import specific submodules
without eagerly loading the full streaming stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "StreamingGateway": (".gateway", "StreamingGateway"),
    "AuthManager": (".auth_manager", "AuthManager"),
    "AdaptiveRateLimiter": (".rate_limiter", "AdaptiveRateLimiter"),
    "ConnectionPool": (".connection_pool", "ConnectionPool"),
    "StreamingHealthMonitor": (".monitoring", "StreamingHealthMonitor"),
    "ProviderConfigValidator": (".providers", "ProviderConfigValidator"),
    "ProviderDiscovery": (".providers", "ProviderDiscovery"),
    "ProviderHealthMonitor": (".providers", "ProviderHealthMonitor"),
    "ProviderRegistry": (".providers", "ProviderRegistry"),
    "StreamingProviderAdapter": (".providers", "StreamingProviderAdapter"),
    "LiveTradingEngine": (".live_trading", "LiveTradingEngine"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
