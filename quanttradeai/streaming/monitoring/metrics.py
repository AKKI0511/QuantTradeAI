"""Prometheus metrics for streaming infrastructure."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram


messages_processed = Counter(
    "websocket_messages_total", "Total processed messages", ["provider", "symbol"]
)
connection_latency = Histogram(
    "websocket_connection_latency_seconds", "Connection establishment time"
)
active_connections = Gauge("websocket_active_connections", "Current active connections")


@dataclass
class Metrics:
    """Convenience wrapper around Prometheus metrics."""

    def record_message(self, provider: str, symbol: str) -> None:
        messages_processed.labels(provider=provider, symbol=symbol).inc()

    def record_connection_latency(self, seconds: float) -> None:
        connection_latency.observe(seconds)

    def set_active_connections(self, count: int) -> None:
        active_connections.set(count)
