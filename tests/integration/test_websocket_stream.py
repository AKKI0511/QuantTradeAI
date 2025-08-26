import asyncio
import json
import tempfile
import yaml
import pandas as pd
from unittest.mock import patch
import pytest

from quanttradeai.data.loader import DataLoader
from quanttradeai.data.datasource import WebSocketDataSource
from quanttradeai.data.processor import DataProcessor


class FakeConnection:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def close(self):
        pass


async def fake_connect(_):
    return FakeConnection([])


@patch("websockets.connect", new=lambda url: fake_connect(url))
def test_streaming_dispatch():
    msg = json.dumps({"symbol": "TEST", "Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1})

    async def connect(_):
        return FakeConnection([msg])

    async def run_test():
        with patch("websockets.connect", new=connect):
            cfg = {"data": {"symbols": ["TEST"], "start_date": "2020-01-01", "end_date": "2020-01-01"}}
            f = tempfile.NamedTemporaryFile("w+", delete=False)
            try:
                yaml.dump(cfg, f)
                f.flush()
                f.close()
                processor = DataProcessor()
                processor.process_data = lambda df: df  # type: ignore
                loader = DataLoader(f.name, data_source=WebSocketDataSource("ws://test"))
                out = []
                await loader.stream_data(processor, callback=lambda df: out.append(df))
                assert len(out) == 1
                pd.testing.assert_frame_equal(out[0], pd.DataFrame([json.loads(msg)]))
            finally:
                try:
                    import os
                    os.unlink(f.name)
                except Exception:
                    pass

    asyncio.run(run_test())

