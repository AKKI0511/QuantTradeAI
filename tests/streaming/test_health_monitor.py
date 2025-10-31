import asyncio
import time
import importlib.util
import pathlib
import sys
import types

monitoring_dir = pathlib.Path(__file__).resolve().parents[2] / "quanttradeai" / "streaming" / "monitoring"
pkg = types.ModuleType("monitoring")
pkg.__path__ = [str(monitoring_dir)]
sys.modules.setdefault("monitoring", pkg)

def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"monitoring.{name}", monitoring_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "monitoring"
    sys.modules[f"monitoring.{name}"] = module
    spec.loader.exec_module(module)
    return module

alerts = _load("alerts")
metrics_collector = _load("metrics_collector")
recovery_manager = _load("recovery_manager")
health = _load("health_monitor")

AlertManager = alerts.AlertManager
MetricsCollector = metrics_collector.MetricsCollector
RecoveryManager = recovery_manager.RecoveryManager
ConnectionHealth = health.ConnectionHealth
StreamingHealthMonitor = health.StreamingHealthMonitor


class CollectingAlertManager(AlertManager):
    def __init__(self):
        super().__init__(channels=["log"])
        self.records = []
        self.callbacks.append(lambda lvl, msg: self.records.append((lvl, msg)))


class DummyRecovery(RecoveryManager):
    def __init__(self):
        super().__init__(max_attempts=1)
        self.called = False

    async def reconnect(self, name: str, connect=None) -> bool:  # pragma: no cover - simple override
        self.called = True
        return True


class CountingRecovery(RecoveryManager):
    def __init__(self):
        super().__init__(max_attempts=3)
        self.calls = 0

    async def reconnect(self, name: str, connect=None) -> bool:  # pragma: no cover - simple override
        self.calls += 1
        return True


class CallbackRecovery(RecoveryManager):
    def __init__(self):
        super().__init__(max_attempts=1)
        self.connect = None
        self.invocations = 0

    async def reconnect(self, name: str, connect=None) -> bool:  # pragma: no cover - simple override
        self.invocations += 1
        self.connect = connect
        if connect is not None:
            await connect()
        return True


async def run_latency_check() -> CollectingAlertManager:
    alerts = CollectingAlertManager()
    monitor = StreamingHealthMonitor(
        connection_status={"c": ConnectionHealth(latency_ms=200)},
        metrics_collector=MetricsCollector(),
        alert_manager=alerts,
        recovery_manager=DummyRecovery(),
        check_interval=0.1,
        thresholds={"max_latency_ms": 100, "min_throughput_msg_per_sec": 0},
    )
    await monitor._check_connections_once()
    return alerts


def test_latency_alert_triggered():
    alerts = asyncio.run(run_latency_check())
    assert any("latency" in msg for _, msg in alerts.records)


async def run_recovery_check() -> DummyRecovery:
    alerts = CollectingAlertManager()
    recovery = DummyRecovery()
    stale = ConnectionHealth()
    stale.last_message_ts = time.time() - 10
    monitor = StreamingHealthMonitor(
        connection_status={"c": stale},
        metrics_collector=MetricsCollector(),
        alert_manager=alerts,
        recovery_manager=recovery,
        check_interval=0.1,
    )
    await monitor._check_connections_once()
    return recovery


def test_stale_connection_triggers_recovery():
    recovery = asyncio.run(run_recovery_check())
    assert recovery.called


def test_last_message_ts_reset_after_reconnect():
    alerts = CollectingAlertManager()
    recovery = CountingRecovery()
    stale = ConnectionHealth()
    stale.last_message_ts = time.time() - 10
    monitor = StreamingHealthMonitor(
        connection_status={"c": stale},
        metrics_collector=MetricsCollector(),
        alert_manager=alerts,
        recovery_manager=recovery,
        check_interval=0.1,
    )
    asyncio.run(monitor._check_connections_once())
    first = recovery.calls
    asyncio.run(monitor._check_connections_once())
    assert recovery.calls == first
    assert stale.last_message_ts > time.time() - 1


def test_recovery_receives_and_invokes_callback():
    alerts = CollectingAlertManager()
    recovery = CallbackRecovery()
    stale = ConnectionHealth()
    stale.last_message_ts = time.time() - 10
    monitor = StreamingHealthMonitor(
        connection_status={"c": stale},
        metrics_collector=MetricsCollector(),
        alert_manager=alerts,
        recovery_manager=recovery,
        check_interval=0.1,
    )

    called = False

    async def reconnect_callback() -> None:
        nonlocal called
        called = True

    monitor.register_connection("c", reconnect_callback=reconnect_callback)
    asyncio.run(monitor._check_connections_once())
    assert recovery.connect is reconnect_callback
    assert called
