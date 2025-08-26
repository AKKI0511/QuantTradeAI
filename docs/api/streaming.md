# Streaming API

QuantTradeAI provides a lightweight, extensible streaming stack for real-time market data. It abstracts provider-specific WebSocket details behind adapters and offers a high-level gateway to manage connections, subscriptions, rate limiting, and monitoring.

## Key Classes

- `StreamingGateway`: High-level orchestrator that reads a YAML config, connects providers, and dispatches messages to callbacks.
- `WebSocketManager`: Manages adapter lifecycle, retries, and circuit breaking.
- `DataProviderAdapter`: Base adapter for providers; Alpaca and Interactive Brokers examples included.
- `AuthManager`, `AdaptiveRateLimiter`, `ConnectionPool`: Opt-in safeguards for production reliability.

## Configuration

File: `config/streaming.yaml`

```yaml
streaming:
  symbols: ["AAPL", "MSFT"]  # global symbols (optional)
  providers:
    - name: "alpaca"
      websocket_url: "wss://stream.data.alpaca.markets/v2/iex"
      auth_method: "api_key"  # expects ALPACA_API_KEY in env
      subscriptions: ["trades", "quotes"]
      # Optional overrides
      # symbols: ["TSLA", "NVDA"]
      rate_limit:
        default_rate: 100
        burst_allowance: 50
      circuit_breaker:
        failure_threshold: 5
        timeout: 30
  buffer_size: 10000
  reconnect_attempts: 5
  health_check_interval: 30
```

Notes:
- If `providers[].symbols` is omitted, `streaming.symbols` is used.
- If no symbols are provided in YAML, subscribe via the API (see below).
- Set `ALPACA_API_KEY` or `${PROVIDER}_API_KEY` in your environment for `auth_method: api_key`.

## Usage

### YAML-Driven

```python
from quanttradeai.streaming import StreamingGateway

gateway = StreamingGateway("config/streaming.yaml")

# Register callbacks (optional if YAML-only subscriptions suffice)
gateway.subscribe_to_trades(["AAPL"], callback=lambda msg: print("trade", msg))
gateway.subscribe_to_quotes(["AAPL"], callback=lambda msg: print("quote", msg))

# Blocking run
# gateway.start_streaming()
```

### Programmatic Subscriptions

```python
from quanttradeai.streaming import StreamingGateway

gw = StreamingGateway("config/streaming.yaml")
gw.subscribe_to_trades(["MSFT", "TSLA"], callback=lambda m: print(m))
# gw.start_streaming()
```

## Extending Providers

Create a new adapter by subclassing `DataProviderAdapter` and implementing `_build_subscribe_message`.

```python
from dataclasses import dataclass
from typing import Any, Dict, List
from quanttradeai.streaming.adapters.base_adapter import DataProviderAdapter

@dataclass
class MyProviderAdapter(DataProviderAdapter):
    name: str = "my_provider"

    def _build_subscribe_message(self, channel: str, symbols: List[str]) -> Dict[str, Any]:
        return {"action": "subscribe", "channel": channel, "symbols": symbols}
```

Register it in your runtime (or fork and extend `AdapterMap` in `StreamingGateway`).

## Monitoring

- Prometheus metrics (`prometheus_client`) track message counts, connection latency, and active connections.
- Optional background health checks ping pooled connections (interval configured via YAML).

