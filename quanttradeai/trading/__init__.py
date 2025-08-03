"""Trading utilities.

Expose portfolio management and risk helper functions.

Public API:
    - :class:`PortfolioManager`
    - :func:`apply_stop_loss_take_profit`
    - :func:`position_size`

Quick Start:
    ```python
    from quanttradeai.trading import PortfolioManager
    pm = PortfolioManager(100000)
    ```
"""

from .portfolio import PortfolioManager
from .risk import apply_stop_loss_take_profit, position_size

__all__ = ["PortfolioManager", "apply_stop_loss_take_profit", "position_size"]
