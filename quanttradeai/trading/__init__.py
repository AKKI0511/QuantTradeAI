"""Trading helpers exposed at the package level."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "PortfolioManager": (".portfolio", "PortfolioManager"),
    "PositionManager": (".position_manager", "PositionManager"),
    "apply_stop_loss_take_profit": (".risk", "apply_stop_loss_take_profit"),
    "position_size": (".risk", "position_size"),
    "DrawdownGuard": (".drawdown_guard", "DrawdownGuard"),
    "RiskManager": (".risk_manager", "RiskManager"),
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
