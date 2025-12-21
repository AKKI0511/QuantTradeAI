import asyncio
from unittest.mock import MagicMock

import pytest

from quanttradeai.streaming.live_trading import LiveTradingEngine
from quanttradeai.streaming.monitoring.health_monitor import StreamingHealthMonitor
from quanttradeai.streaming.stream_buffer import StreamBuffer


class FakeGateway:
    def __init__(self) -> None:
        self.buffer = StreamBuffer(16)
        self.health_monitor = StreamingHealthMonitor()

    async def _start(self) -> None:
        return None


class DummyProcessor:
    def process_data(self, df):
        out = df.copy()
        out["f1"] = 1.0
        return out


class DummyModel:
    def __init__(self, outputs=None) -> None:
        self.feature_columns = ["f1"]
        self.outputs = outputs or [1]
        self.model = True

    def load_model(self, path: str) -> None:  # pragma: no cover - noop for tests
        return None

    def predict(self, X):
        value = self.outputs.pop(0) if self.outputs else 0
        return [value for _ in range(len(X))]


@pytest.mark.asyncio
async def test_live_trading_engine_opens_position_and_records_metrics():
    gateway = FakeGateway()
    latency_spy = MagicMock()
    queue_depth_spy = MagicMock()
    gateway.health_monitor.metrics_collector.record_latency = latency_spy
    gateway.health_monitor.metrics_collector.record_queue_depth = queue_depth_spy

    engine = LiveTradingEngine(
        model_config="config/model_config.yaml",
        model_path="unused",
        gateway=gateway,
        data_processor=DummyProcessor(),
        model=DummyModel(outputs=[1]),
        min_history_for_features=1,
        history_window=4,
        risk_config=None,
        position_manager_config=None,
    )

    consumer = asyncio.create_task(engine._consume_buffer())
    await gateway.buffer.put(
        {
            "symbol": "AAPL",
            "price": 10.0,
            "Open": 10.0,
            "High": 10.0,
            "Low": 10.0,
            "Close": 10.0,
            "Volume": 100,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )
    await asyncio.sleep(0.05)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert engine.execution_log
    assert engine.execution_log[0]["action"] == "buy"
    latency_spy.assert_called()
    queue_depth_spy.assert_called()


@pytest.mark.asyncio
async def test_live_trading_engine_closes_position_on_sell_signal():
    gateway = FakeGateway()
    engine = LiveTradingEngine(
        model_config="config/model_config.yaml",
        model_path="unused",
        gateway=gateway,
        data_processor=DummyProcessor(),
        model=DummyModel(outputs=[1, -1]),
        min_history_for_features=1,
        history_window=8,
        risk_config=None,
        position_manager_config=None,
    )

    consumer = asyncio.create_task(engine._consume_buffer())
    await gateway.buffer.put(
        {
            "symbol": "MSFT",
            "price": 50.0,
            "Open": 50.0,
            "High": 50.0,
            "Low": 50.0,
            "Close": 50.0,
            "Volume": 100,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )
    await gateway.buffer.put(
        {
            "symbol": "MSFT",
            "price": 51.0,
            "Open": 51.0,
            "High": 51.0,
            "Low": 51.0,
            "Close": 51.0,
            "Volume": 100,
            "timestamp": "2024-01-01T00:01:00Z",
        }
    )
    await asyncio.sleep(0.05)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    actions = [entry["action"] for entry in engine.execution_log]
    assert actions == ["buy", "sell"]
    assert "MSFT" not in engine.portfolio.positions
