"""Model implementations.

This package currently provides the :class:`MomentumClassifier` ensemble
for generating trading labels.

Public API:
    - :class:`MomentumClassifier`

Quick Start:
    ```python
    from quanttradeai.models import MomentumClassifier
    model = MomentumClassifier()
    ```
"""

from .classifier import MomentumClassifier

__all__ = ["MomentumClassifier"]
