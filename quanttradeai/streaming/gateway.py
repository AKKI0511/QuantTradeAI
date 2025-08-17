"""High-level streaming gateway orchestrator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import yaml

from .stream_buffer import StreamBuffer
from .websocket_manager import WebSocketManager
from .adapters.alpaca_adapter import AlpacaAdapter
from .adapters.ib_adapter import IBAdapter
from .processors import MessageProcessor
from .monitoring.metrics import Metrics

AdapterMap = {
    "alpaca": AlpacaAdapter,
    "interactive_brokers": IBAdapter,
}

Callback = Callable[[str, Dict], None]


@dataclass
class StreamingGateway:
    """Main entry point for real-time data streaming."""

    config_path: str
    websocket_manager: WebSocketManager = field(init=False)
    message_processor: MessageProcessor = field(init=False)
    buffer: StreamBuffer = field(init=False)
    metrics: Metrics = field(init=False)
    _subscriptions: List[Tuple[str, List[str]]] = field(default_factory=list)
    _callbacks: Dict[str, List[Callback]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)["streaming"]
        self.websocket_manager = WebSocketManager(
            reconnect_attempts=cfg.get("reconnect_attempts", 5)
        )
        self.message_processor = MessageProcessor()
        self.buffer = StreamBuffer(cfg.get("buffer_size", 1000))
        self.metrics = Metrics()
        for provider_cfg in cfg.get("providers", []):
            name = provider_cfg["name"]
            url = provider_cfg["websocket_url"]
            adapter_cls = AdapterMap.get(name)
            if adapter_cls is None:
                raise ValueError(f"Unknown provider: {name}")
            adapter = adapter_cls(websocket_url=url)
            self.websocket_manager.add_adapter(adapter)

    # Subscription API -------------------------------------------------
    def subscribe_to_trades(self, symbols: List[str], callback: Callback) -> None:
        """Subscribe to trade updates for ``symbols``."""

        self._subscriptions.append(("trades", symbols))
        self._callbacks.setdefault("trades", []).append(callback)

    def subscribe_to_quotes(self, symbols: List[str], callback: Callback) -> None:
        """Subscribe to quote updates for ``symbols``."""

        self._subscriptions.append(("quotes", symbols))
        self._callbacks.setdefault("quotes", []).append(callback)

    # Runtime ----------------------------------------------------------
    async def _dispatch(self, provider: str, message: Dict) -> None:
        self.metrics.increment()
        processed = self.message_processor.process(message)
        await self.buffer.put(processed)
        msg_type = message.get("type", "trades")
        for cb in self._callbacks.get(msg_type, []):
            res = cb(processed)
            if asyncio.iscoroutine(res):
                await res

    async def _start(self) -> None:
        await self.websocket_manager.connect_all()
        for channel, symbols in self._subscriptions:
            for adapter in self.websocket_manager.adapters:
                await adapter.subscribe(channel, symbols)
        await self.websocket_manager.run(self._dispatch)

    def start_streaming(self) -> None:
        """Blocking call that starts the streaming event loop."""

        asyncio.run(self._start())
