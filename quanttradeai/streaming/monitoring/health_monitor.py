from __future__ import annotations

"""Streaming connection health monitoring."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from .alerts import AlertManager
from .metrics_collector import MetricsCollector
from .recovery_manager import RecoveryManager


@dataclass
class ConnectionHealth:
    """State information about a single streaming connection."""

    last_message_ts: float = field(default_factory=lambda: time.time())
    messages: int = 0
    bytes_received: int = 0
    window_start: float = field(default_factory=lambda: time.time())
    latency_ms: float = 0.0
    status: str = "connected"
    reconnect_attempts: int = 0
    last_sequence: Optional[int] = None
    lost_messages: int = 0

    def record_message(
        self, sequence: Optional[int] = None, size_bytes: Optional[int] = None
    ) -> None:
        self.messages += 1
        self.last_message_ts = time.time()
        if size_bytes is not None:
            self.bytes_received += size_bytes
        if sequence is not None:
            if self.last_sequence is not None and sequence > self.last_sequence + 1:
                self.lost_messages += sequence - self.last_sequence - 1
            self.last_sequence = sequence


@dataclass
class StreamingHealthMonitor:
    """Monitor streaming connections and collect metrics."""

    connection_status: Dict[str, ConnectionHealth] = field(default_factory=dict)
    metrics_collector: MetricsCollector = field(default_factory=MetricsCollector)
    alert_manager: AlertManager = field(default_factory=AlertManager)
    recovery_manager: RecoveryManager = field(default_factory=RecoveryManager)
    check_interval: float = 5.0
    thresholds: Dict[str, float] = field(default_factory=dict)
    queue_size_fn: Optional[Callable[[], int]] = None
    queue_name: str = "stream"
    _running: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    def register_connection(self, name: str) -> None:
        self.connection_status.setdefault(name, ConnectionHealth())

    def record_message(
        self,
        name: str,
        *,
        sequence: Optional[int] = None,
        size_bytes: Optional[int] = None,
    ) -> None:
        if name not in self.connection_status:
            self.register_connection(name)
        self.connection_status[name].record_message(sequence, size_bytes)

    def record_latency(self, name: str, latency_ms: float) -> None:
        if name not in self.connection_status:
            self.register_connection(name)
        self.connection_status[name].latency_ms = latency_ms

    # ------------------------------------------------------------------
    async def monitor_connection_health(self) -> None:
        self._running = True
        while self._running:
            await self._check_connections_once()
            await asyncio.sleep(self.check_interval)

    async def _check_connections_once(self) -> None:
        now = time.time()
        for name, health in self.connection_status.items():
            elapsed = now - health.window_start
            throughput = health.messages / elapsed if elapsed > 0 else 0.0
            bandwidth = health.bytes_received / elapsed if elapsed > 0 else 0.0
            age = now - health.last_message_ts
            self.metrics_collector.record_throughput(name, throughput)
            self.metrics_collector.record_bandwidth(name, bandwidth)
            self.metrics_collector.record_data_freshness(name, age)
            if health.latency_ms:
                self.metrics_collector.record_latency(name, health.latency_ms)

            max_lat = self.thresholds.get("max_latency_ms")
            if max_lat is not None and health.latency_ms > max_lat:
                self.trigger_alerts("warning", f"{name} latency {health.latency_ms}ms")

            min_tp = self.thresholds.get("min_throughput_msg_per_sec")
            if min_tp is not None and throughput < min_tp:
                self.trigger_alerts(
                    "warning", f"{name} throughput {throughput:.2f}/s below threshold"
                )

            if health.lost_messages:
                self.metrics_collector.increment_message_loss(
                    name, health.lost_messages
                )
                self.trigger_alerts(
                    "warning", f"{name} lost {health.lost_messages} messages"
                )
                health.lost_messages = 0

            # Detect stale connections
            if age > self.check_interval * 2:
                await self.handle_connection_failure(name, health)

            health.window_start = now
            health.messages = 0
            health.bytes_received = 0

        if self.queue_size_fn is not None:
            depth = self.queue_size_fn()
            self.metrics_collector.record_queue_depth(self.queue_name, depth)
            max_depth = self.thresholds.get("max_queue_depth")
            if max_depth is not None and depth > max_depth:
                self.trigger_alerts(
                    "warning", f"queue depth {depth} exceeds {max_depth}"
                )

    async def handle_connection_failure(
        self, name: str, health: ConnectionHealth
    ) -> None:
        health.status = "reconnecting"
        health.reconnect_attempts += 1
        self.metrics_collector.increment_reconnect(name)
        success = await self.recovery_manager.reconnect(name)
        if success:
            health.status = "connected"
            health.window_start = time.time()
            health.messages = 0
        else:
            health.status = "down"
            self.trigger_alerts("error", f"{name} reconnection failed")
        max_attempts = self.thresholds.get("max_reconnect_attempts")
        if max_attempts is not None and health.reconnect_attempts > max_attempts:
            self.trigger_alerts(
                "error",
                f"{name} exceeded max reconnect attempts: {health.reconnect_attempts}",
            )

    def collect_performance_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        try:  # best effort; psutil may not be installed
            import psutil

            proc = psutil.Process()
            metrics["memory_bytes"] = float(proc.memory_info().rss)
            metrics["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        except Exception:  # pragma: no cover - optional dependency
            pass
        return metrics

    def trigger_alerts(self, level: str, message: str) -> None:
        self.alert_manager.send(level, message)

    def generate_health_report(self) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for name, health in self.connection_status.items():
            report[name] = {
                "status": health.status,
                "latency_ms": health.latency_ms,
                "messages": health.messages,
                "reconnect_attempts": health.reconnect_attempts,
            }
        return report
