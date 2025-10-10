from __future__ import annotations

import pytest

from quanttradeai.streaming.adapters.example_provider import ExampleStreamingProvider
from quanttradeai.streaming.providers.base import ProviderCapabilities


@pytest.mark.asyncio
async def test_example_provider_lifecycle() -> None:
    adapter = ExampleStreamingProvider()
    await adapter.connect()
    assert adapter.is_connected is True

    await adapter.subscribe(["AAPL", "BTCUSD"])
    assert adapter.subscriptions == {"AAPL", "BTCUSD"}

    await adapter.unsubscribe(["AAPL"])
    assert adapter.subscriptions == {"BTCUSD"}

    await adapter.disconnect()
    assert adapter.is_connected is False


def test_example_provider_capabilities() -> None:
    adapter = ExampleStreamingProvider()
    capabilities = adapter.get_capabilities()
    assert isinstance(capabilities, ProviderCapabilities)
    assert capabilities.supports_order_book is True
    assert capabilities.rate_limit_per_minute == 1200
    assert capabilities.max_subscriptions == 500


def test_example_provider_validate_config_defaults() -> None:
    adapter = ExampleStreamingProvider()
    normalized = adapter.validate_config({"options": {"mode": "demo"}})
    assert "options" in normalized
    assert "credentials" in normalized
    assert normalized["options"]["mode"] == "demo"
