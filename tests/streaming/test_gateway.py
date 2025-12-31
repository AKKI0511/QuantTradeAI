import asyncio
import json
import tempfile
import time
import warnings
from typing import Awaitable, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import yaml

from quanttradeai.streaming.monitoring import StreamingHealthMonitor

import pytest
from quanttradeai.streaming import StreamingGateway

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*websockets.*:DeprecationWarning",
    "ignore:.*WebSocketServerProtocol is deprecated:DeprecationWarning",
)

warnings.filterwarnings(
    "ignore",
    message=r"websockets\.legacy is deprecated",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"websockets\.server\.WebSocketServerProtocol is deprecated",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"websockets.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"uvicorn\.protocols\.websockets.*",
)


class StubProviderMonitor:
    def __init__(
        self,
        *,
        streaming_monitor: Optional[StreamingHealthMonitor] = None,
        **_: Dict,
    ) -> None:
        self.streaming_monitor = streaming_monitor or StreamingHealthMonitor()
        self.recovery_manager = self.streaming_monitor.recovery_manager
        self.registered: List[str] = []
        self.failover_handlers: Dict[str, Callable[[], Awaitable[None]]] = {}
        self.execute_calls: List[str] = []
        self.record_success_calls: List[float] = []
        self.record_failure_calls: List[str] = []
        self.status_providers: Dict[str, Callable[[], object]] = {}

    def register_provider(
        self,
        provider_name: str,
        *,
        failover_handler: Optional[Callable[[], Awaitable[None]]] = None,
        status_provider: Optional[Callable[[], object]] = None,
    ) -> None:
        self.streaming_monitor.register_connection(provider_name)
        self.registered.append(provider_name)
        if failover_handler is not None:
            self.failover_handlers[provider_name] = failover_handler
        if status_provider is not None:
            self.status_providers[provider_name] = status_provider

    async def execute_with_health(
        self,
        provider_name: str,
        operation: Callable[[], Awaitable[object]],
        *,
        fallback: Optional[Callable[[], Awaitable[object]]] = None,
    ) -> object:
        self.execute_calls.append(provider_name)
        start = time.perf_counter()
        try:
            result = await operation()
        except Exception as exc:
            await self.record_failure(provider_name, exc)
            if fallback is not None:
                return await fallback()
            raise
        latency_ms = (time.perf_counter() - start) * 1000.0
        await self.record_success(provider_name, latency_ms)
        return result

    async def record_success(
        self, provider_name: str, latency_ms: float, *, bytes_received: int = 0
    ) -> None:
        self.record_success_calls.append(latency_ms)

    async def record_failure(self, provider_name: str, error: Exception) -> None:
        self.record_failure_calls.append(provider_name)


class FakeConnection:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def close(self):
        pass


def test_gateway_streaming():
    msg = json.dumps({"type": "trades", "symbol": "TEST", "price": 1})

    async def connect(url, *_, **__):
        return FakeConnection([msg])

    async def run_test():
        with patch("websockets.connect", new=connect):
            cfg = {
                "streaming": {
                    "providers": [
                        {
                            "name": "alpaca",
                            "websocket_url": "ws://test",
                            "auth_method": "none",
                            "subscriptions": ["trades"],
                        }
                    ]
                }
            }
            f = tempfile.NamedTemporaryFile("w+", delete=False)
            try:
                yaml.safe_dump(cfg, f)
                f.flush()
                f.close()
                gateway = StreamingGateway(f.name)
                out = []
                gateway.subscribe_to_trades(["TEST"], callback=lambda data: out.append(data))
                await gateway._start()
                assert out == [{"type": "trades", "symbol": "TEST", "price": 1}]
            finally:
                try:
                    import os
                    os.unlink(f.name)
                except Exception:
                    pass

    asyncio.run(run_test())


@patch("quanttradeai.streaming.gateway.ProviderHealthMonitor", new=StubProviderMonitor)
def test_gateway_registers_providers_and_failover(tmp_path):
    cfg = {
        "streaming": {
            "providers": [
                {
                    "name": "alpaca",
                    "websocket_url": "ws://test",
                    "auth_method": "none",
                }
            ]
        }
    }
    config_file = tmp_path / "streaming.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    gateway = StreamingGateway(str(config_file))
    monitor = gateway.provider_monitor
    assert monitor.registered == ["alpaca"]
    adapter = gateway.websocket_manager.adapters[0]
    gateway.websocket_manager._connect_with_retry = AsyncMock()
    failover = monitor.failover_handlers[adapter.name]
    asyncio.run(failover())
    gateway.websocket_manager._connect_with_retry.assert_awaited_once()
    _, kwargs = gateway.websocket_manager._connect_with_retry.await_args
    assert kwargs["monitor"] is monitor


@patch("quanttradeai.streaming.gateway.ProviderHealthMonitor", new=StubProviderMonitor)
def test_gateway_start_uses_provider_monitor(tmp_path):
    cfg = {
        "streaming": {
            "providers": [
                {
                    "name": "alpaca",
                    "websocket_url": "ws://test",
                    "auth_method": "none",
                }
            ]
        }
    }
    config_file = tmp_path / "streaming.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    gateway = StreamingGateway(str(config_file))
    adapter = gateway.websocket_manager.adapters[0]
    gateway.subscribe_to_trades(["TEST"], callback=lambda _: None)
    adapter.subscribe = AsyncMock(return_value=None)
    gateway.websocket_manager.connect_all = AsyncMock()
    gateway.websocket_manager.run = AsyncMock()
    gateway.health_monitor.monitor_connection_health = AsyncMock()

    async def run_start():
        await gateway._start()

    asyncio.run(run_start())

    gateway.websocket_manager.connect_all.assert_awaited_once()
    _, kwargs = gateway.websocket_manager.connect_all.await_args
    assert kwargs["monitor"] is gateway.provider_monitor
    adapter.subscribe.assert_awaited_once()
    assert gateway.provider_monitor.execute_calls.count("alpaca") == len(
        gateway._subscriptions
    )


@patch("quanttradeai.streaming.gateway.ProviderHealthMonitor", new=StubProviderMonitor)
def test_gateway_registers_reconnect_callback(tmp_path):
    cfg = {
        "streaming": {
            "providers": [
                {
                    "name": "alpaca",
                    "websocket_url": "ws://test",
                    "auth_method": "none",
                }
            ]
        }
    }
    config_file = tmp_path / "streaming.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    gateway = StreamingGateway(str(config_file))
    adapter = gateway.websocket_manager.adapters[0]
    gateway.websocket_manager.connect_all = AsyncMock()
    gateway.websocket_manager.run = AsyncMock()
    gateway.websocket_manager._connect_with_retry = AsyncMock()
    adapter.subscribe = AsyncMock()
    gateway.health_monitor.monitor_connection_health = AsyncMock()

    async def run_start():
        await gateway._start()

    asyncio.run(run_start())

    callback = gateway.health_monitor.reconnect_callbacks[adapter.name]
    asyncio.run(callback())
    gateway.websocket_manager._connect_with_retry.assert_awaited_with(
        adapter, monitor=gateway.provider_monitor
    )


@patch("quanttradeai.streaming.gateway.ProviderHealthMonitor", new=StubProviderMonitor)
@patch("quanttradeai.streaming.gateway.start_http_server")
def test_metrics_exporter_starts_when_enabled(start_http_server, tmp_path):
    cfg = {
        "streaming": {
            "providers": [
                {
                    "name": "alpaca",
                    "websocket_url": "ws://test",
                    "auth_method": "none",
                }
            ],
            "health_check_interval": 0,
        },
        "streaming_health": {
            "metrics": {"enabled": True, "host": "127.0.0.1", "port": 9100}
        },
    }
    config_file = tmp_path / "streaming.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    gateway = StreamingGateway(str(config_file))
    adapter = gateway.websocket_manager.adapters[0]
    adapter.subscribe = AsyncMock()
    gateway.websocket_manager.connect_all = AsyncMock()
    gateway.websocket_manager.run = AsyncMock()
    gateway.health_monitor.monitor_connection_health = AsyncMock()

    async def run_start():
        await gateway._start()

    asyncio.run(run_start())

    start_http_server.assert_called_once_with(9100, addr="127.0.0.1")


@patch("quanttradeai.streaming.gateway.ProviderHealthMonitor", new=StubProviderMonitor)
@patch("quanttradeai.streaming.gateway.start_http_server")
def test_metrics_exporter_skipped_when_disabled(start_http_server, tmp_path):
    cfg = {
        "streaming": {
            "providers": [
                {
                    "name": "alpaca",
                    "websocket_url": "ws://test",
                    "auth_method": "none",
                }
            ],
            "health_check_interval": 0,
        },
        "streaming_health": {"metrics": {"enabled": False}},
    }
    config_file = tmp_path / "streaming.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    gateway = StreamingGateway(str(config_file))
    adapter = gateway.websocket_manager.adapters[0]
    adapter.subscribe = AsyncMock()
    gateway.websocket_manager.connect_all = AsyncMock()
    gateway.websocket_manager.run = AsyncMock()
    gateway.health_monitor.monitor_connection_health = AsyncMock()

    async def run_start():
        await gateway._start()

    asyncio.run(run_start())

    start_http_server.assert_not_called()


@patch("quanttradeai.streaming.gateway.ProviderHealthMonitor", new=StubProviderMonitor)
@patch("quanttradeai.streaming.gateway.start_http_server")
def test_metrics_exporter_skips_when_sharing_api_port(start_http_server, tmp_path):
    cfg = {
        "streaming": {
            "providers": [
                {
                    "name": "alpaca",
                    "websocket_url": "ws://test",
                    "auth_method": "none",
                }
            ],
            "health_check_interval": 0,
        },
        "streaming_health": {
            "api": {"enabled": True, "host": "0.0.0.0", "port": 9000},
            "metrics": {"enabled": True, "host": "0.0.0.0", "port": 9000},
        },
    }
    config_file = tmp_path / "streaming.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    gateway = StreamingGateway(str(config_file))
    adapter = gateway.websocket_manager.adapters[0]
    adapter.subscribe = AsyncMock()
    gateway.websocket_manager.connect_all = AsyncMock()
    gateway.websocket_manager.run = AsyncMock()
    gateway.health_monitor.monitor_connection_health = AsyncMock()

    async def run_start():
        await gateway._start()

    asyncio.run(run_start())

    start_http_server.assert_not_called()


@patch("quanttradeai.streaming.gateway.ProviderHealthMonitor", new=StubProviderMonitor)
def test_websocket_manager_reports_failures(tmp_path):
    cfg = {
        "streaming": {
            "providers": [
                {
                    "name": "alpaca",
                    "websocket_url": "ws://test",
                    "auth_method": "none",
                }
            ]
        }
    }
    config_file = tmp_path / "streaming.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    gateway = StreamingGateway(str(config_file))
    monitor = gateway.provider_monitor
    manager = gateway.websocket_manager
    adapter = manager.adapters[0]
    manager.reconnect_attempts = 1

    async def failing_connect():
        raise RuntimeError("boom")

    adapter.connect = AsyncMock(side_effect=failing_connect)

    async def run_connect():
        await manager._connect_with_retry(adapter, monitor=monitor)

    with pytest.raises(RuntimeError):
        asyncio.run(run_connect())
    assert monitor.record_failure_calls
