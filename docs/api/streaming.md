# Streaming API

QuantTradeAI ships with a plugin-ready streaming provider framework. Providers are implemented as adapters, discovered at runtime, validated against explicit configuration models, and wrapped with health monitoring utilities for capability negotiation and failover handling. The existing `StreamingGateway` continues to orchestrate WebSocket connections defined in `config/streaming.yaml`, while the provider subsystem enables richer lifecycle control when integrating third-party feeds.

## Key Modules

- **`StreamingGateway`** – High-level orchestrator that reads `config/streaming.yaml`, manages built-in adapters, and dispatches normalized messages to callbacks.
- **`StreamingProviderAdapter`** – Abstract base class for provider integrations. Implement connection, subscription, and capability reporting logic.
- **`ProviderDiscovery`** – Scans adapter packages (with hot reload support) and registers available providers in a `ProviderRegistry`.
- **`ProviderRegistry`** – Stores `ProviderMetadata`, performs dependency validation, and instantiates adapters on demand.
- **`ProviderConfigValidator`** – Validates provider-specific configuration files and negotiates runtime settings against advertised capabilities.
- **`ProviderHealthMonitor`** – Extends the streaming health monitor with per-provider metrics, circuit breaking, and failover hooks.

## Provider Configuration

Providers define their own configuration documents (YAML or JSON) that describe supported environments. Example (`config/providers/example.yaml`):

```yaml
provider: example
environment: dev
environments:
  dev:
    asset_types: ["stocks", "crypto"]
    data_types: ["trades", "quotes", "order_book"]
    rate_limit_per_minute: 1200
    max_subscriptions: 500
    options:
      mode: realtime
    credentials:
      token: ${API_TOKEN}
```

Use `ProviderConfigValidator` to load and validate this file while resolving environment variables and capability constraints:

```python
from quanttradeai.streaming.adapters.example_provider import ExampleStreamingProvider
from quanttradeai.streaming.providers import ProviderConfigValidator

validator = ProviderConfigValidator()
model = validator.load_from_path("config/providers/example.yaml", environment="dev")
adapter = ExampleStreamingProvider()
runtime = validator.validate(adapter, model)  # raises on unsupported assets, limits, or auth
```

## Runtime Discovery and Hot Reload

```python
from quanttradeai.streaming.providers import ProviderDiscovery

discovery = ProviderDiscovery()      # defaults to quanttradeai.streaming.adapters
registry = discovery.discover()

for metadata in registry.list():
    print(metadata.name, metadata.version, metadata.capabilities.asset_types)

# After dropping a new adapter module into quanttradeai/streaming/adapters/
registry = discovery.refresh()       # hot reload; new providers are registered automatically
```

The registry enforces dependency availability and version precedence to prevent conflicting registrations.

## Health Monitoring and Failover

```python
import asyncio

from quanttradeai.streaming.providers import ProviderHealthMonitor

monitor = ProviderHealthMonitor(error_threshold=5)
monitor.register_provider(
    adapter.provider_name,
    status_provider=adapter.get_health_status,
    failover_handler=lambda: adapter.connect(),
)

async def run() -> None:
    await monitor.execute_with_health(adapter.provider_name, adapter.connect)
    await monitor.execute_with_health(
        adapter.provider_name,
        lambda: adapter.subscribe(["AAPL"]),
    )
    status = monitor.get_status(adapter.provider_name)
    print(status.status, status.latency_ms)

asyncio.run(run())
```

`ProviderHealthMonitor` wraps adapter operations with circuit breakers, rolling error windows, and optional failover callbacks coordinated through the shared recovery manager.

## Integrating with `StreamingGateway`

For existing YAML-driven workflows, continue to instantiate `StreamingGateway("config/streaming.yaml")` and register callbacks via `subscribe_to_trades`/`subscribe_to_quotes`. Custom providers discovered at runtime can be wired into the gateway by extending its adapter map or by running sidecar tasks that publish normalized events into the same processing pipeline.

## Health API Configuration

`StreamingGateway` exposes the original health monitoring controls via the `streaming_health` section in `config/streaming.yaml`. The provider-centric monitor shares the same recovery manager and metrics subsystem, so alert thresholds, Prometheus integration, and REST endpoints (`/health`, `/status`, `/metrics`) continue to operate as documented in the gateway README.


