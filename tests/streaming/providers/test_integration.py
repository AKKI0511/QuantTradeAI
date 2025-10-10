from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quanttradeai.streaming.providers import (
    ProviderConfigValidator,
    ProviderDiscovery,
    ProviderHealthMonitor,
)


@pytest.mark.asyncio
async def test_provider_integration_end_to_end(tmp_path: Path) -> None:
    discovery = ProviderDiscovery()
    registry = discovery.discover()
    adapter = registry.create_instance("example")

    config_path = tmp_path / "example.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "provider": "example",
                "environment": "dev",
                "environments": {
                    "dev": {
                        "asset_types": ["stocks"],
                        "data_types": ["trades", "quotes"],
                        "options": {"mode": "realtime"},
                    }
                },
            },
            handle,
        )

    validator = ProviderConfigValidator()
    model = validator.load_from_path(config_path, environment="dev")
    runtime = validator.validate(adapter, model, environment="dev")
    assert runtime.provider == "example"

    monitor = ProviderHealthMonitor(error_threshold=5)
    monitor.register_provider(
        adapter.provider_name, status_provider=adapter.get_health_status
    )

    await monitor.execute_with_health(adapter.provider_name, adapter.connect)
    assert adapter.is_connected is True

    await adapter.subscribe(["AAPL"])
    await monitor.record_success(adapter.provider_name, latency_ms=1.2, bytes_received=42)
    status = monitor.get_status(adapter.provider_name)
    assert status.status == "connected"
    assert status.metrics["subscriptions"] >= 1

    async def failing_operation() -> None:
        raise ValueError("boom")

    async def fallback_operation() -> str:
        return "fallback"

    result = await monitor.execute_with_health(
        adapter.provider_name,
        failing_operation,
        fallback=fallback_operation,
    )
    assert result == "fallback"
    assert monitor.get_status(adapter.provider_name).status in {"error", "circuit_open"}

    await adapter.disconnect()
