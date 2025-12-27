"""High-level streaming gateway orchestrator.

Example
-------
>>> # Basic usage with YAML-only configuration
>>> # config/streaming.yaml defines providers, subscriptions and symbols
>>> from quanttradeai.streaming import StreamingGateway
>>> gw = StreamingGateway("config/streaming.yaml")
>>> # Register a simple callback for trade messages
>>> gw.subscribe_to_trades(["AAPL"], callback=lambda m: print(m))
>>> # Start the event loop (blocking)
>>> # gw.start_streaming()

>>> # Programmatic subscriptions (no YAML changes needed)
>>> gw = StreamingGateway("config/streaming.yaml")
>>> gw.subscribe_to_quotes(["MSFT", "TSLA"], callback=lambda m: print(m))
>>> # gw.start_streaming()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import yaml
from prometheus_client import start_http_server

from .stream_buffer import StreamBuffer
from .websocket_manager import WebSocketManager
from .adapters.alpaca_adapter import AlpacaAdapter
from .adapters.base_adapter import DataProviderAdapter
from .adapters.ib_adapter import IBAdapter
from .processors import MessageProcessor
from .monitoring import (
    AlertManager,
    MetricsCollector,
    RecoveryManager,
    StreamingHealthMonitor,
    create_health_app,
)
from .monitoring.metrics import Metrics
from .logging import logger
from .providers import ProviderHealthMonitor

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
    health_monitor: StreamingHealthMonitor = field(init=False)
    provider_monitor: ProviderHealthMonitor = field(init=False)
    _subscriptions: List[Tuple[str, List[str]]] = field(default_factory=list)
    _callbacks: Dict[str, List[Callback]] = field(default_factory=dict)
    _config_symbols: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        with open(self.config_path, "r") as f:
            cfg_all = yaml.safe_load(f)
            cfg = cfg_all.get("streaming", {})
        self.websocket_manager = WebSocketManager(
            reconnect_attempts=cfg.get("reconnect_attempts", 5)
        )
        self.message_processor = MessageProcessor()
        self.buffer = StreamBuffer(cfg.get("buffer_size", 1000))
        self.metrics = Metrics()
        health_cfg = cfg_all.get("streaming_health", {})
        mon_cfg = health_cfg.get("monitoring", {})
        thresh_cfg = health_cfg.get("thresholds", {})
        alert_cfg = health_cfg.get("alerts", {})
        self.health_monitor = StreamingHealthMonitor(
            metrics_collector=MetricsCollector(),
            alert_manager=AlertManager(
                alert_cfg.get("channels", ["log"]),
                escalation_threshold=alert_cfg.get("escalation_threshold", 3),
            ),
            recovery_manager=RecoveryManager(
                max_attempts=thresh_cfg.get("max_reconnect_attempts", 5),
                reset_timeout=thresh_cfg.get("circuit_breaker_timeout", 60),
            ),
            check_interval=mon_cfg.get("check_interval", 5),
            thresholds={
                "max_latency_ms": thresh_cfg.get("max_latency_ms", 100),
                "min_throughput_msg_per_sec": thresh_cfg.get(
                    "min_throughput_msg_per_sec", 0
                ),
                "max_reconnect_attempts": thresh_cfg.get("max_reconnect_attempts", 5),
                "max_queue_depth": thresh_cfg.get("max_queue_depth", 0),
            },
            queue_size_fn=self.buffer.queue.qsize,
            queue_name="stream",
        )
        self.provider_monitor = ProviderHealthMonitor(
            streaming_monitor=self.health_monitor
        )
        api_cfg = health_cfg.get("api", {})
        self._api_enabled = api_cfg.get("enabled", False)
        self._api_host = api_cfg.get("host", "0.0.0.0")
        self._api_port = api_cfg.get("port", 8000)
        metrics_cfg = health_cfg.get("metrics", {})
        self._metrics_enabled = metrics_cfg.get("enabled", False)
        self._metrics_host = metrics_cfg.get("host", "0.0.0.0")
        self._metrics_port = metrics_cfg.get("port", 9000)
        # Top-level symbol list for convenience
        self._config_symbols = cfg.get("symbols", []) or []
        for provider_cfg in cfg.get("providers", []):
            name = provider_cfg["name"]
            url = provider_cfg["websocket_url"]
            adapter_cls = AdapterMap.get(name)
            if adapter_cls is None:
                raise ValueError(f"Unknown provider: {name}")
            adapter = adapter_cls(websocket_url=url)
            self.websocket_manager.add_adapter(
                adapter,
                circuit_breaker_cfg=provider_cfg.get("circuit_breaker", {}),
                rate_limit_cfg=provider_cfg.get("rate_limit"),
                auth_method=provider_cfg.get("auth_method", "api_key"),
            )

            async def _failover(adapter=adapter) -> None:
                await self.websocket_manager._connect_with_retry(
                    adapter,
                    monitor=self.provider_monitor,
                )

            self.provider_monitor.register_provider(
                adapter.name,
                failover_handler=_failover,
            )
            # Config-driven subscriptions (optional)
            subs = provider_cfg.get("subscriptions", []) or []
            prov_symbols = provider_cfg.get("symbols", self._config_symbols) or []
            for channel in subs:
                # Store once; adapters subscribe per provider in _start
                self._subscriptions.append((channel, prov_symbols))

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
    async def _apply_subscriptions(self, adapter: DataProviderAdapter) -> None:
        for channel, symbols in self._subscriptions:

            async def _subscribe(
                adapter=adapter,
                channel=channel,
                symbols=symbols,
            ) -> None:
                await adapter.subscribe(channel, symbols)

            await self.provider_monitor.execute_with_health(
                adapter.name,
                _subscribe,
            )

    async def _dispatch(self, provider: str, message: Dict) -> None:
        symbol = message.get("symbol", "")
        self.metrics.record_message(provider, symbol)
        seq = message.get("sequence")
        size = len(str(message))
        self.health_monitor.record_message(provider, sequence=seq, size_bytes=size)
        processed = self.message_processor.process(message)
        await self.buffer.put(processed)
        msg_type = message.get("type", "trades")
        logger.debug(
            "dispatch_message", provider=provider, type=msg_type, symbol=symbol
        )
        for cb in self._callbacks.get(msg_type, []):
            try:
                res = cb(processed)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:  # keep streaming robust to callback errors
                logger.error("callback_error", error=str(exc))

    async def _start(self) -> None:
        await self.websocket_manager.connect_all(monitor=self.provider_monitor)
        for adapter in self.websocket_manager.adapters:

            async def _reconnect_adapter(adapter=adapter) -> None:
                await self.websocket_manager._connect_with_retry(
                    adapter, monitor=self.provider_monitor
                )
                await self._apply_subscriptions(adapter)

            self.health_monitor.register_connection(
                adapter.name, reconnect_callback=_reconnect_adapter
            )
        asyncio.create_task(self.health_monitor.monitor_connection_health())
        metrics_port_bound_to_api = (
            self._api_enabled
            and self._metrics_host == self._api_host
            and self._metrics_port == self._api_port
        )
        if self._metrics_enabled and not metrics_port_bound_to_api:
            try:
                start_http_server(self._metrics_port, addr=self._metrics_host)
            except Exception:
                logger.warning("metrics_exporter_start_failed")
        if self._api_enabled:
            try:
                import uvicorn

                app = create_health_app(self.health_monitor)
                config = uvicorn.Config(
                    app, host=self._api_host, port=self._api_port, log_level="warning"
                )
                server = uvicorn.Server(config)
                asyncio.create_task(server.serve())
            except Exception:
                logger.warning("health_api_start_failed")
        # Optional legacy health check for connection pool
        try:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f).get("streaming", {})
            interval = int(cfg.get("health_check_interval", 0))
        except Exception:
            interval = 0
        if interval and interval > 0:
            asyncio.create_task(
                self.websocket_manager.connection_pool.health_check_loop(interval)
            )
        for adapter in self.websocket_manager.adapters:
            await self._apply_subscriptions(adapter)
        await self.websocket_manager.run(self._dispatch)

    def start_streaming(self) -> None:
        """Blocking call that starts the streaming event loop."""

        asyncio.run(self._start())
