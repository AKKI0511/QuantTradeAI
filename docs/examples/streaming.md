# Real-Time Streaming Examples

The streaming subsystem now supports plugin-based providers with configuration validation, discovery, and health-aware execution. The snippets below demonstrate how to work with the reference `ExampleStreamingProvider`. Replace the provider name and configuration with your implementation as needed.

## Prerequisites

- Install dependencies (via Poetry or pip)
- Export any credentials referenced in provider configs, for example:

```bash
export API_TOKEN="super-secret"
```

Create `config/providers/example.yaml`:

```yaml
provider: example
environment: dev
environments:
  dev:
    asset_types: ["stocks", "crypto"]
    data_types: ["trades", "quotes"]
    credentials:
      token: ${API_TOKEN}
    options:
      mode: realtime
```

## Example 1: Discover and Validate a Provider

```python
from quanttradeai.streaming.providers import ProviderConfigValidator, ProviderDiscovery

discovery = ProviderDiscovery()
registry = discovery.discover()
adapter = registry.create_instance("example")

validator = ProviderConfigValidator()
model = validator.load_from_path("config/providers/example.yaml", environment="dev")
runtime = validator.validate(adapter, model)

print(runtime.asset_types)  # {'stocks', 'crypto'}
# runtime.options / runtime.credentials contain the normalized settings for your adapter
```

## Example 2: Connect with Health Monitoring

Continue from Example 1 using the same `adapter` instance.

```python
import asyncio

from quanttradeai.streaming.providers import ProviderHealthMonitor

monitor = ProviderHealthMonitor(error_threshold=3)
monitor.register_provider(
    adapter.provider_name,
    status_provider=adapter.get_health_status,
)

async def main() -> None:
    await monitor.execute_with_health(adapter.provider_name, adapter.connect)
    await monitor.execute_with_health(
        adapter.provider_name,
        lambda: adapter.subscribe(["AAPL", "MSFT"]),
    )
    status = monitor.get_status(adapter.provider_name)
    print(status.status, status.metrics["subscriptions"])

asyncio.run(main())
```

## Example 3: Hot Reload Newly Added Providers

```python
# Continuing with the same discovery instance from Example 1
registry = discovery.refresh()  # clears caches and re-imports modules
print([metadata.name for metadata in registry.list()])
```

## Example 4: Fallback Handling

Continue from Example 2 with the configured `monitor`.

```python
import asyncio

async def fallback() -> str:
    return "fallback-value"

async def main() -> None:
    result = await monitor.execute_with_health(
        adapter.provider_name,
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),  # force an error
        fallback=fallback,
    )
    print(result)  # "fallback-value"

asyncio.run(main())
```

## Integrating with the Gateway

- Continue to configure `config/streaming.yaml` for built-in adapters and shared buffering.
- Custom providers can publish into the same processing pipeline by extending `StreamingGateway`'s adapter map or streaming events alongside the gateway.
- Health API, metrics, and alerting remain enabled through the existing `streaming_health` section in `config/streaming.yaml`.

