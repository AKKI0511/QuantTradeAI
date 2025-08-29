from __future__ import annotations

"""Prometheus-based metrics collection for streaming health."""

from dataclasses import dataclass
from prometheus_client import Counter, Gauge, REGISTRY


def _get_metric(metric_cls, name: str, documentation: str, labelnames: list[str]):
    try:
        return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]
    except KeyError:
        return metric_cls(name, documentation, labelnames)


# Metric definitions -----------------------------------------------------------
_throughput = _get_metric(
    Gauge,
    "stream_message_throughput_per_sec",
    "Incoming message rate per connection",
    ["connection"],
)
_latency = _get_metric(
    Gauge,
    "stream_connection_latency_ms",
    "Measured ping-pong latency in ms",
    ["connection"],
)
_freshness = _get_metric(
    Gauge,
    "stream_data_freshness_seconds",
    "Seconds since the last message was received",
    ["connection"],
)
_reconnects = _get_metric(
    Counter,
    "stream_reconnect_total",
    "Total reconnection attempts",
    ["connection"],
)


@dataclass
class MetricsCollector:
    """Lightweight wrapper around Prometheus metrics."""

    def record_throughput(self, connection: str, rate: float) -> None:
        _throughput.labels(connection=connection).set(rate)

    def record_latency(self, connection: str, latency_ms: float) -> None:
        _latency.labels(connection=connection).set(latency_ms)

    def record_data_freshness(self, connection: str, age: float) -> None:
        _freshness.labels(connection=connection).set(age)

    def increment_reconnect(self, connection: str) -> None:
        _reconnects.labels(connection=connection).inc()
