"""Data acquisition and preparation.

Provides utilities to download market data and transform it into model
ready datasets.

Public API:
    - :class:`DataLoader`
    - :class:`DataProcessor`
    - :class:`DataSource` implementations

Quick Start:
    ```python
    from quanttradeai.data import DataLoader
    loader = DataLoader()
    data = loader.fetch_data()
    ```
"""

from .loader import DataLoader
from .processor import DataProcessor
from .datasource import (
    DataSource,
    YFinanceDataSource,
    AlphaVantageDataSource,
    WebSocketDataSource,
)

__all__ = [
    "DataLoader",
    "DataProcessor",
    "DataSource",
    "YFinanceDataSource",
    "AlphaVantageDataSource",
    "WebSocketDataSource",
]
