import asyncio
import time
from datetime import datetime

import pytest
from pybreaker import CircuitBreakerError

from quanttradeai.streaming.auth_manager import AuthManager
from quanttradeai.streaming.connection_pool import ConnectionPool
from quanttradeai.streaming.rate_limiter import AdaptiveRateLimiter
from quanttradeai.streaming.websocket_manager import WebSocketManager
from quanttradeai.streaming.adapters.base_adapter import DataProviderAdapter


class FailingAdapter(DataProviderAdapter):
    async def _build_subscribe_message(self, channel, symbols):
        return {}

    async def connect(self):
        raise RuntimeError("fail")


def test_circuit_breaker_opens():
    adapter = FailingAdapter(websocket_url="ws://test", name="fail")
    manager = WebSocketManager(reconnect_attempts=1)
    manager.add_adapter(adapter, circuit_breaker_cfg={"failure_threshold": 1, "timeout": 1})

    async def run():
        with pytest.raises(CircuitBreakerError):
            await manager.connect_all()

    asyncio.run(run())


def test_rate_limiter_enforces_rate():
    limiter = AdaptiveRateLimiter(base_rate=1, burst_size=1)

    async def run():
        start = time.perf_counter()
        await asyncio.gather(limiter.acquire(), limiter.acquire())
        return time.perf_counter() - start

    elapsed = asyncio.run(run())
    assert elapsed >= 1


def test_auth_manager_refresh(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "KEY123")
    am = AuthManager("alpaca")
    am._expires_at = datetime.utcnow()  # force refresh
    headers = asyncio.run(am.get_auth_headers())
    assert headers["Authorization"] == "Bearer KEY123"


def test_connection_pool_reuses_connections():
    pool = ConnectionPool(max_connections=1)

    async def factory():
        return object()

    async def run():
        conn1 = await pool.acquire_connection(factory)
        await pool.release(conn1)
        conn2 = await pool.acquire_connection(factory)
        assert conn1 is conn2

    asyncio.run(run())
