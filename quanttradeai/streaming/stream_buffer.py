"""Async buffer for streaming messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from asyncio import Queue
from typing import Any


@dataclass
class StreamBuffer:
    """Thread-safe buffer based on :class:`asyncio.Queue`.

    Parameters
    ----------
    maxsize:
        Maximum number of messages to buffer.
    """

    maxsize: int
    queue: Queue = field(init=False)

    def __post_init__(self) -> None:
        self.queue = Queue(maxsize=self.maxsize)

    async def put(self, item: Any) -> None:
        """Put a message into the buffer."""

        await self.queue.put(item)

    async def get(self) -> Any:
        """Retrieve a message from the buffer."""

        return await self.queue.get()
