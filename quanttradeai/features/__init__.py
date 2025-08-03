"""Feature engineering.

Collection of technical and custom indicator helpers used to generate
model features.

Public API:
    - :mod:`technical`
    - :mod:`custom`

Quick Start:
    ```python
    from quanttradeai.features import technical as ta
    ta.sma(close, 20)
    ```
"""

from . import technical, custom

__all__ = ["technical", "custom"]
