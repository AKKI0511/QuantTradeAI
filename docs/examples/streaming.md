# Real-Time Streaming Examples

This page shows end-to-end streaming examples using the built-in Alpaca adapter. Replace the URL and credentials to match your provider.

## Prerequisites

- Install package dependencies (Poetry or pip)
- Export provider API key, for example:

```bash
export ALPACA_API_KEY="<your-key>"
```

## Example 1: YAML-Only Subscriptions

`config/streaming.yaml`:

```yaml
streaming:
  symbols: ["AAPL"]
  providers:
    - name: "alpaca"
      websocket_url: "wss://stream.data.alpaca.markets/v2/iex"
      auth_method: "api_key"
      subscriptions: ["trades", "quotes"]
  buffer_size: 1000
  reconnect_attempts: 3
  health_check_interval: 30
```

Python:

```python
from quanttradeai.streaming import StreamingGateway

gw = StreamingGateway("config/streaming.yaml")

def on_trade(msg):
    print("TRADE", msg)

def on_quote(msg):
    print("QUOTE", msg)

gw.subscribe_to_trades(["AAPL"], on_trade)
gw.subscribe_to_quotes(["AAPL"], on_quote)
# gw.start_streaming()
```

## Example 2: Programmatic Symbols

```python
from quanttradeai.streaming import StreamingGateway

gw = StreamingGateway("config/streaming.yaml")
gw.subscribe_to_trades(["MSFT", "TSLA"], lambda m: print(m))
# gw.start_streaming()
```

## Example 3: Custom Adapter

```python
from dataclasses import dataclass
from typing import Any, Dict, List
from quanttradeai.streaming.adapters.base_adapter import DataProviderAdapter
from quanttradeai.streaming.websocket_manager import WebSocketManager

@dataclass
class MyAdapter(DataProviderAdapter):
    name: str = "custom"
    def _build_subscribe_message(self, channel: str, symbols: List[str]) -> Dict[str, Any]:
        return {"action": "subscribe", channel: symbols}

manager = WebSocketManager()
manager.add_adapter(MyAdapter(websocket_url="wss://example"), auth_method="none")
# Use StreamingGateway or call manager.connect_all()/run() directly.
```

## Health Monitoring Configuration

Append the following to `config/streaming.yaml` to enable comprehensive health checks:

```yaml
streaming_health:
  monitoring:
    enabled: true
    check_interval: 5
  thresholds:
    max_latency_ms: 100
    min_throughput_msg_per_sec: 50
    max_queue_depth: 5000
    circuit_breaker_timeout: 60
  alerts:
    enabled: true
    channels: ["log", "metrics"]
    escalation_threshold: 3
  api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
```

Start the gateway and query metrics:

```bash
curl http://localhost:8000/status
```

## Alert Threshold Tuning

- Increase `max_latency_ms` or decrease `escalation_threshold` for noisy networks.
- Monitor `max_queue_depth` during peak sessions and adjust to avoid drops.
- Use Prometheus metrics to derive realistic throughput baselines.

## Production Deployment Recommendations

- Run the health API behind an authenticated ingress when exposing publicly.
- Scrape `/metrics` with Prometheus and forward alerts to your incident system.
- Configure fallback providers to ensure continuity during outages.

