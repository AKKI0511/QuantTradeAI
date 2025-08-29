import asyncio
import time

from fastapi.testclient import TestClient

from quanttradeai.streaming.monitoring import (
    AlertManager,
    MetricsCollector,
    RecoveryManager,
    StreamingHealthMonitor,
    create_health_app,
)


class CollectingAlerts(AlertManager):
    def __init__(self, **kwargs):
        super().__init__(channels=["log"], **kwargs)
        self.records = []
        self.callbacks.append(lambda lvl, msg: self.records.append((lvl, msg)))


def test_message_loss_and_queue_depth():
    alerts = CollectingAlerts(escalation_threshold=3)
    monitor = StreamingHealthMonitor(
        metrics_collector=MetricsCollector(),
        alert_manager=alerts,
        recovery_manager=RecoveryManager(max_attempts=1),
        thresholds={"max_queue_depth": 10},
        queue_size_fn=lambda: 20,
    )
    monitor.record_message("c", sequence=1)
    monitor.record_message("c", sequence=3)
    asyncio.run(monitor._check_connections_once())
    assert any("lost" in m for _, m in alerts.records)
    assert any("queue depth" in m for _, m in alerts.records)


def test_alert_escalation():
    alerts = CollectingAlerts(escalation_threshold=2)
    alerts.send("warning", "issue")
    alerts.send("warning", "issue")
    assert ("error", "issue") in alerts.records


def test_health_api_endpoints():
    monitor = StreamingHealthMonitor()
    app = create_health_app(monitor)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/status").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_recovery_manager_circuit_breaker():
    rec = RecoveryManager(max_attempts=1, reset_timeout=1)

    async def bad_connect():
        raise RuntimeError("fail")

    assert not asyncio.run(rec.reconnect("c", bad_connect))
    assert not asyncio.run(rec.reconnect("c", bad_connect))
