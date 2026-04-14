import pandas as pd
import pytest

from quanttradeai.streaming.replay import ReplayGateway


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    index = pd.to_datetime([timestamp for timestamp, _price in rows], utc=True)
    prices = [price for _timestamp, price in rows]
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [price + 1.0 for price in prices],
            "Low": [price - 1.0 for price in prices],
            "Close": prices,
            "Volume": [100.0 + offset for offset, _ in enumerate(prices)],
        },
        index=index,
    )


@pytest.mark.asyncio
async def test_replay_gateway_emits_one_bar_per_row_in_time_order():
    gateway = ReplayGateway(
        frames={
            "MSFT": _frame(
                [
                    ("2024-01-02T00:00:00Z", 210.0),
                    ("2024-01-03T00:00:00Z", 211.0),
                ]
            ),
            "AAPL": _frame(
                [
                    ("2024-01-01T00:00:00Z", 100.0),
                    ("2024-01-03T00:00:00Z", 101.0),
                ]
            ),
        },
        pace_delay_ms=0,
        buffer_size=16,
    )

    await gateway._start()

    messages = []
    while not gateway.buffer.queue.empty():
        messages.append(await gateway.buffer.get())

    assert len(messages) == 4
    assert [message["type"] for message in messages] == ["replay_bar"] * 4
    assert [message["symbol"] for message in messages] == [
        "AAPL",
        "MSFT",
        "AAPL",
        "MSFT",
    ]
    assert [message["timestamp"] for message in messages] == [
        "2024-01-01T00:00:00+00:00",
        "2024-01-02T00:00:00+00:00",
        "2024-01-03T00:00:00+00:00",
        "2024-01-03T00:00:00+00:00",
    ]
    assert messages[0]["open"] == pytest.approx(100.0)
    assert messages[-1]["close"] == pytest.approx(211.0)
    assert messages[-1]["volume"] == pytest.approx(101.0)
