import pandas as pd
import pytest

from quanttradeai.streaming.history import ReplayWindow, split_replay_frames
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

    assert len(messages) == 5
    assert [message["type"] for message in messages] == [
        "replay_bar",
        "replay_bar",
        "replay_bar",
        "replay_bar",
        "replay_complete",
    ]
    assert [message["symbol"] for message in messages[:-1]] == [
        "AAPL",
        "MSFT",
        "AAPL",
        "MSFT",
    ]
    assert [message["timestamp"] for message in messages[:-1]] == [
        "2024-01-01T00:00:00+00:00",
        "2024-01-02T00:00:00+00:00",
        "2024-01-03T00:00:00+00:00",
        "2024-01-03T00:00:00+00:00",
    ]
    assert messages[0]["open"] == pytest.approx(100.0)
    assert messages[-2]["close"] == pytest.approx(211.0)
    assert messages[-2]["volume"] == pytest.approx(101.0)


def test_split_replay_frames_includes_intraday_rows_on_end_date():
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [101.0, 102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.5, 101.5, 102.5, 103.5],
            "Volume": [100.0, 110.0, 120.0, 130.0],
        },
        index=pd.to_datetime(
            [
                "2024-02-01T15:30:00Z",
                "2024-02-02T09:30:00Z",
                "2024-02-02T15:30:00Z",
                "2024-02-03T09:30:00Z",
            ],
            utc=True,
        ),
    )

    bootstrap_frames, replay_frames, manifest = split_replay_frames(
        {"AAPL": frame},
        replay_window=ReplayWindow(
            start_date="2024-02-02",
            end_date="2024-02-02",
            pace_delay_ms=0,
        ),
        history_window=10,
    )

    assert list(bootstrap_frames["AAPL"].index) == [
        pd.Timestamp("2024-02-01T15:30:00Z")
    ]
    assert list(replay_frames["AAPL"].index) == [
        pd.Timestamp("2024-02-02T09:30:00Z"),
        pd.Timestamp("2024-02-02T15:30:00Z"),
    ]
    assert manifest["symbols"][0]["replay_end"] == "2024-02-02T15:30:00+00:00"
