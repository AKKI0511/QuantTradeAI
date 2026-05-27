# Streaming API

## Overview

The streaming API provides realtime ingestion, deterministic replay, provider abstraction, buffering, and health monitoring for paper/live agent runs. These components are importable Python APIs, but many are orchestration internals. Prefer direct use when you are embedding QuantTradeAI into another async system, writing provider adapters, or testing paper/live behavior.

## Public Imports

```python
from quanttradeai.streaming import (
    StreamingGateway,
    ReplayGateway,
    LiveTradingEngine,
    AuthManager,
    AdaptiveRateLimiter,
    ConnectionPool,
    StreamingHealthMonitor,
    ProviderConfigValidator,
    ProviderDiscovery,
    ProviderHealthMonitor,
    ProviderRegistry,
    StreamingProviderAdapter,
)

from quanttradeai import WebSocketDataSource
```

`WebSocketDataSource` is exposed from the top-level package and from `quanttradeai.data.datasource`; the higher-level gateway classes live under `quanttradeai.streaming`.

## Main Classes and Functions

| API | Import Path | Purpose |
| --- | --- | --- |
| `StreamingGateway` | `quanttradeai.streaming.gateway` | High-level realtime gateway from streaming YAML |
| `WebSocketDataSource` | `quanttradeai.data.datasource` | Lightweight generic WebSocket data source |
| `ReplayGateway` | `quanttradeai.streaming.replay` | Replay historical bars into a stream buffer |
| `LiveTradingEngine` | `quanttradeai.streaming.live_trading` | Coordinate streaming, features, model inference, risk, and execution |
| `ReplayWindow` | `quanttradeai.streaming.history` | Paper replay date window |
| History helpers | `quanttradeai.streaming.history` | Timeframe parsing, UTC index normalization, replay splitting |
| `WebSocketManager` | `quanttradeai.streaming.websocket_manager` | Adapter connection lifecycle |
| `StreamBuffer` | `quanttradeai.streaming.stream_buffer` | Async queue wrapper |
| Provider APIs | `quanttradeai.streaming.providers` | Adapter base classes, discovery, registry, config validation |
| Monitoring APIs | `quanttradeai.streaming.monitoring` | Health reports, alerts, metrics, recovery |

## `StreamingGateway`

**Signature**

```python
@dataclass
class StreamingGateway:
    config_path: str

    def subscribe_to_trades(self, symbols: list[str], callback: Callback) -> None
    def subscribe_to_quotes(self, symbols: list[str], callback: Callback) -> None
    def start_streaming(self) -> None
```

`StreamingGateway` reads a runtime streaming YAML file, creates provider adapters, applies configured subscriptions, dispatches normalized messages through a `StreamBuffer`, and can expose metrics/health endpoints when configured.

```python
from quanttradeai.streaming import StreamingGateway

gateway = StreamingGateway("config/streaming.yaml")
gateway.subscribe_to_trades(["AAPL"], callback=lambda message: print(message))
gateway.start_streaming()  # blocking
```

Although the `Callback` alias is defined in the module, the current dispatcher invokes callbacks with one processed message argument.

**Current provider names**

The gateway adapter map currently recognizes:

| Provider Name | Adapter |
| --- | --- |
| `alpaca` | `quanttradeai.streaming.adapters.alpaca_adapter.AlpacaAdapter` |
| `interactive_brokers` | `quanttradeai.streaming.adapters.ib_adapter.IBAdapter` |

An unknown provider name raises `ValueError`.

## `WebSocketDataSource`

**Signature**

```python
class WebSocketDataSource(DataSource):
    def __init__(self, url: str) -> None
    async def connect(self) -> None
    async def subscribe(self, symbols: list[str]) -> None
    async def stream(self) -> AsyncIterator[dict]
    async def close(self) -> None
```

Use this when you only need a simple JSON WebSocket source. It is lower-level than `StreamingGateway` and does not provide provider adapters, health monitoring, or buffering.

```python
from quanttradeai import WebSocketDataSource

source = WebSocketDataSource("wss://example.test/feed")
await source.subscribe(["AAPL"])
async for message in source.stream():
    print(message)
    break
await source.close()
```

## Replay and History Helpers

### `ReplayGateway`

**Signature**

```python
@dataclass
class ReplayGateway:
    frames: dict[str, pd.DataFrame]
    pace_delay_ms: int = 0
    buffer_size: int = 1000
```

`ReplayGateway` converts historical OHLCV frames into messages shaped like streaming bars:

```python
{
    "type": "replay_bar",
    "symbol": symbol,
    "timestamp": timestamp.isoformat(),
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "price": float,
    "volume": float,
}
```

It is primarily used by paper agent runs and by tests that inject a deterministic gateway into `LiveTradingEngine`.

```python
from quanttradeai.streaming import ReplayGateway

gateway = ReplayGateway({"AAPL": bars}, pace_delay_ms=0)
```

### `ReplayWindow`

**Signature**

```python
@dataclass(frozen=True)
class ReplayWindow:
    start_date: str
    end_date: str
    pace_delay_ms: int = 0
```

### History Functions

| Function | Signature | Purpose |
| --- | --- | --- |
| `parse_iso_date` | `(value: str, *, field_name: str) -> date` | Validate ISO `YYYY-MM-DD` dates |
| `parse_timeframe` | `(timeframe: str) -> tuple[int, str] | None` | Parse timeframe strings such as `1m`, `1h`, `1d`, `1wk`, `1mo` |
| `timeframe_to_pandas_freq` | `(timeframe: str) -> str` | Convert to pandas frequency |
| `bootstrap_window_delta` | `(timeframe: str, bars: int) -> timedelta` | Estimate bootstrap lookback |
| `bucket_for_timestamp` | `(timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp` | Bucket timestamps to timeframe |
| `ensure_utc_datetime_index` | `(df: pd.DataFrame) -> pd.DataFrame` | Return UTC-indexed sorted frame |
| `build_streaming_runtime_model_config` | `(model_cfg, *, bootstrap_bars, end_date=None, replay_start_date=None, now=None) -> dict` | Expand model data window for streaming bootstrap |
| `seed_history_frames` | `(frames, *, history_window) -> dict[str, pd.DataFrame]` | Keep recent OHLCV bars per symbol |
| `split_replay_frames` | `(frames, *, replay_window, history_window) -> tuple[dict, dict, dict]` | Split bootstrap and replay frames |

```python
from quanttradeai.streaming.history import ReplayWindow, split_replay_frames

bootstrap, replay, manifest = split_replay_frames(
    frames,
    replay_window=ReplayWindow("2024-03-01", "2024-03-31"),
    history_window=220,
)
```

## `LiveTradingEngine`

**Signature**

```python
@dataclass
class LiveTradingEngine:
    model_config: str
    model_path: str
    features_config: str = "config/features_config.yaml"
    streaming_config: str = "config/streaming.yaml"
    risk_config: str | None = "config/risk_config.yaml"
    position_manager_config: str | None = "config/position_manager.yaml"
    enable_health_api: bool | None = None
    initial_capital: float = 1_000_000.0
    max_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.10
    history_window: int = 512
    min_history_for_features: int = 220
    stop_loss_pct: float = 0.01
    shutdown_drain_timeout: float = 0.5
    execution_hook: Callable[[dict], None] | None = None
    gateway: StreamingGateway | ReplayGateway | None = None
    data_processor: DataProcessor | None = None
    model: MomentumClassifier | None = None
    bootstrap_history_frames: dict[str, pd.DataFrame] | None = None
    execution_backend: str = "simulated"
```

**Key methods and properties**

| API | Purpose |
| --- | --- |
| `broker_provider` | Provider name when a broker runtime is attached |
| `health_monitor` | Gateway health monitor if available |
| `bootstrap_history()` | Load initial history from `bootstrap_history_frames` or `DataLoader` |
| `start()` | Async start method for gateway and inference |
| `execution_log` | In-memory execution payloads |
| `decision_log` | In-memory signal/action payloads |

`LiveTradingEngine` loads a saved `MomentumClassifier`, builds features on rolling history, predicts a signal, applies risk checks, and records executions.

```python
from quanttradeai.streaming import LiveTradingEngine

engine = LiveTradingEngine(
    model_config="config/model_config.yaml",
    features_config="config/features_config.yaml",
    streaming_config="config/streaming.yaml",
    model_path="models/example",
)

await engine.start()
```

## WebSocket Manager and Buffer

### `WebSocketManager`

**Signature**

```python
@dataclass
class WebSocketManager:
    reconnect_attempts: int = 5
    adapters: list[DataProviderAdapter] = field(default_factory=list)
    connection_pool: ConnectionPool = field(default_factory=ConnectionPool)

    def add_adapter(... ) -> None
    async def connect_all(self, *, monitor: ProviderHealthMonitor | None = None) -> None
    async def run(self, callback: Callable[[str, dict], Awaitable[None]]) -> None
```

Use it directly when building a custom streaming gateway.

### `StreamBuffer`

**Signature**

```python
@dataclass
class StreamBuffer:
    maxsize: int
    async def put(self, item: Any) -> None
    async def get(self) -> Any
```

Wraps an `asyncio.Queue`.

```python
from quanttradeai.streaming.stream_buffer import StreamBuffer

buffer = StreamBuffer(maxsize=1000)
await buffer.put({"symbol": "AAPL", "price": 180.0})
message = await buffer.get()
```

## Provider APIs

### `StreamingProviderAdapter`

**Signature**

```python
class StreamingProviderAdapter(ABC):
    provider_name: str = "base"
    provider_version: str = "0.0.0"
    provider_description: str = ""
    provider_dependencies: Sequence[str] = ()
    default_capabilities: ProviderCapabilities = ProviderCapabilities()

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None
    async def connect(self) -> None
    async def disconnect(self) -> None
    async def subscribe(self, symbols: Sequence[str]) -> None
    async def unsubscribe(self, symbols: Sequence[str]) -> None
    def get_capabilities(self) -> ProviderCapabilities
    def validate_config(self, config: Mapping[str, Any]) -> MutableMapping[str, Any]
    def get_health_status(self) -> ProviderHealthStatus
    @classmethod
    def metadata(cls) -> dict[str, Any]
```

Use this base class for pluggable provider integrations discovered by `ProviderDiscovery`.

### Provider Utility Classes

| API | Purpose |
| --- | --- |
| `ProviderCapabilities` | Supported asset/data types, limits, auth requirement |
| `MarketDataEvent`, `QuoteEvent`, `TradeEvent`, `OrderBookEvent` | Normalized event dataclasses |
| `ProviderHealthStatus` | Provider health snapshot |
| `ProviderRegistry` | Register, list, and instantiate provider adapters |
| `ProviderDiscovery` | Discover adapter classes from provider packages or paths |
| `ProviderConfigValidator` | Load/validate YAML or JSON provider config and resolve env vars |

```python
from quanttradeai.streaming import ProviderDiscovery

registry = ProviderDiscovery().discover()
provider_names = [metadata.name for metadata in registry.list()]
```

## Monitoring APIs

### `StreamingHealthMonitor`

**Signature**

```python
@dataclass
class StreamingHealthMonitor:
    def register_connection(self, name: str, reconnect_callback=None) -> None
    def record_message(self, name: str, *, sequence=None, size_bytes=None) -> None
    def record_latency(self, name: str, latency_ms: float) -> None
    async def monitor_connection_health(self) -> None
    async def handle_connection_failure(self, name: str, health: ConnectionHealth) -> None
    def collect_performance_metrics(self) -> dict[str, float]
    def trigger_alerts(self, level: str, message: str) -> None
    def generate_health_report(self) -> dict[str, dict[str, float]]
```

```python
from quanttradeai.streaming import StreamingHealthMonitor

monitor = StreamingHealthMonitor()
monitor.register_connection("alpaca")
monitor.record_message("alpaca", sequence=1, size_bytes=256)
report = monitor.generate_health_report()
```

### Other Monitoring Components

| API | Purpose |
| --- | --- |
| `ConnectionHealth` | Per-connection state and message counters |
| `AlertManager` | Alert dispatch with log/metrics/callback support |
| `MetricsCollector` | Prometheus metrics for throughput, latency, freshness, reconnects, queue depth |
| `RecoveryManager` | Exponential-backoff reconnect attempts with circuit breaking |
| `ProviderHealthMonitor` | Provider-specific health, failover, and fallback execution |
| `create_health_app(monitor)` | FastAPI app exposing `/health`, `/status`, and `/metrics` |

## Minimal Examples

### Realtime Gateway Callback

```python
from quanttradeai.streaming import StreamingGateway

def on_trade(message: dict) -> None:
    print(message["symbol"], message.get("price"))

gateway = StreamingGateway("config/streaming.yaml")
gateway.subscribe_to_trades(["AAPL"], on_trade)
```

### Replay Window Split

```python
from quanttradeai.streaming.history import ReplayWindow, split_replay_frames

bootstrap, replay, manifest = split_replay_frames(
    frames,
    replay_window=ReplayWindow(start_date="2024-01-15", end_date="2024-02-15"),
    history_window=220,
)
```

## Relationship to Paper/Live Agent Runs

Paper and live agent runners compile project streaming settings, construct gateways, and then run `LiveTradingEngine` or rule/LLM/hybrid streaming engines. Replay-backed paper mode uses `ReplayGateway`; realtime paper/live mode uses `StreamingGateway`.

## Related CLI/YAML Docs

- [CLI agent docs](../cli/agents.md)
- [Streaming config](../config/data-and-streaming.md)
- [Streaming examples](../examples/streaming.md)
- [Artifacts](../artifacts.md)

## Common Mistakes

- Calling `StreamingGateway.start_streaming()` inside an already-running event loop; use lower-level async methods or an engine in async contexts.
- Expecting `ReplayGateway` to connect to a provider; it only replays DataFrames into a buffer.
- Passing lowercase OHLCV history into replay without checking downstream consumers; replay emits both lowercase message fields and `price`, but feature generation still expects title-case history frames.
- Using a provider name not present in the current `StreamingGateway` adapter map.
- Enabling health API and metrics on the same host/port without confirming the resulting server behavior in your deployment.
