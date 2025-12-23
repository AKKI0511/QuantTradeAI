import asyncio
from unittest.mock import MagicMock

import pytest

from quanttradeai.streaming.live_trading import LiveTradingEngine
from quanttradeai.streaming.monitoring.health_monitor import StreamingHealthMonitor
from quanttradeai.streaming.stream_buffer import StreamBuffer
from quanttradeai.trading.position_manager import PositionManager


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
async def test_live_trading_engine_normalizes_provider_payloads():
    gateway = FakeGateway()
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
            "S": "TSLA",
            "p": "123.45",
            "t": "2024-01-01T00:00:00Z",
            "v": 10,
        }
    )
    await asyncio.sleep(0.05)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert engine.execution_log
    assert engine.execution_log[0]["symbol"] == "TSLA"
    assert "TSLA" in engine._history
    assert engine._history["TSLA"]["Close"].iloc[-1] == pytest.approx(123.45)


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


@pytest.mark.asyncio
async def test_live_trading_engine_orders_and_deduplicates_history():
    gateway = FakeGateway()
    engine = LiveTradingEngine(
        model_config="config/model_config.yaml",
        model_path="unused",
        gateway=gateway,
        data_processor=DummyProcessor(),
        model=DummyModel(outputs=[1, 1, 1]),
        min_history_for_features=1,
        history_window=3,
        risk_config=None,
        position_manager_config=None,
    )

    consumer = asyncio.create_task(engine._consume_buffer())
    await gateway.buffer.put(
        {
            "symbol": "AAPL",
            "price": 101.0,
            "Open": 101.0,
            "High": 101.0,
            "Low": 101.0,
            "Close": 101.0,
            "Volume": 50,
            "timestamp": "2024-01-01T00:01:00Z",
        }
    )
    await gateway.buffer.put(
        {
            "symbol": "AAPL",
            "price": 100.0,
            "Open": 100.0,
            "High": 100.0,
            "Low": 100.0,
            "Close": 100.0,
            "Volume": 40,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )
    await gateway.buffer.put(
        {
            "symbol": "AAPL",
            "price": 102.0,
            "Open": 102.0,
            "High": 102.0,
            "Low": 102.0,
            "Close": 102.0,
            "Volume": 60,
            "timestamp": "2024-01-01T00:02:00Z",
        }
    )
    await gateway.buffer.put(
        {
            "symbol": "AAPL",
            "price": 103.0,
            "Open": 103.0,
            "High": 103.0,
            "Low": 103.0,
            "Close": 103.0,
            "Volume": 70,
            "timestamp": "2024-01-01T00:02:00Z",
        }
    )
    await asyncio.sleep(0.05)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    history = engine._history["AAPL"]
    assert list(history.index) == sorted(history.index)
    assert len(history) == 3  # deduped window, keeps last duplicate
    assert history["Close"].iloc[-1] == pytest.approx(103.0)


@pytest.mark.asyncio
async def test_position_manager_cash_mirrors_portfolio_without_double_counting():
    gateway = FakeGateway()
    position_manager = PositionManager()
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
    engine.position_manager = position_manager
    engine.position_manager.cash = engine.portfolio.cash

    consumer = asyncio.create_task(engine._consume_buffer())
    await gateway.buffer.put(
        {
            "symbol": "GOOG",
            "price": 100.0,
            "Open": 100.0,
            "High": 100.0,
            "Low": 100.0,
            "Close": 100.0,
            "Volume": 10,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )
    await gateway.buffer.put(
        {
            "symbol": "GOOG",
            "price": 101.0,
            "Open": 101.0,
            "High": 101.0,
            "Low": 101.0,
            "Close": 101.0,
            "Volume": 12,
            "timestamp": "2024-01-01T00:01:00Z",
        }
    )
    await asyncio.sleep(0.05)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert engine.position_manager is not None
    assert engine.position_manager.cash == pytest.approx(engine.portfolio.cash)


@pytest.mark.asyncio
async def test_live_trading_engine_marks_to_market_and_updates_risk():
    gateway = FakeGateway()
    risk_spy = MagicMock()
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
    engine.risk_manager.update = risk_spy
    engine.portfolio.positions["AAPL"] = {
        "qty": 1,
        "price": 10.0,
        "stop_loss_pct": 0.01,
    }

    consumer = asyncio.create_task(engine._consume_buffer())
    await gateway.buffer.put(
        {
            "symbol": "AAPL",
            "price": 11.0,
            "Open": 11.0,
            "High": 11.0,
            "Low": 11.0,
            "Close": 11.0,
            "Volume": 10,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )
    await asyncio.sleep(0.05)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert engine.portfolio.positions["AAPL"]["price"] == pytest.approx(11.0)
    risk_spy.assert_called()


@pytest.mark.asyncio
async def test_live_trading_engine_skips_invalid_price_updates():
    gateway = FakeGateway()
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
            "symbol": "NFLX",
            "price": 0,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )
    await asyncio.sleep(0.05)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert "NFLX" not in engine._history


@pytest.mark.asyncio
async def test_live_trading_engine_skips_non_finite_price_updates():
    gateway = FakeGateway()
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
    risk_spy = MagicMock()
    engine.risk_manager.update = risk_spy
    engine.portfolio.positions["AAPL"] = {
        "qty": 1,
        "price": 10.0,
        "stop_loss_pct": 0.01,
    }

    consumer = asyncio.create_task(engine._consume_buffer())
    await gateway.buffer.put(
        {
            "symbol": "AAPL",
            "price": float("nan"),
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )
    await asyncio.sleep(0.05)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert engine.portfolio.positions["AAPL"]["price"] == pytest.approx(10.0)
    assert "AAPL" not in engine._history
    risk_spy.assert_not_called()
