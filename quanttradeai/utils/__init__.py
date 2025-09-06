"""Miscellaneous utilities.

Expose common metrics lazily to avoid heavy optional dependencies.

Public API:
    - :mod:`metrics`
"""

__all__ = ["metrics"]


def __getattr__(name: str):
    if name == "metrics":
        from . import metrics as _metrics

        return _metrics
    raise AttributeError(name)
