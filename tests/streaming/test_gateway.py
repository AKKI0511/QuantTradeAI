import asyncio
import json
import tempfile
from unittest.mock import patch

import yaml

from quanttradeai.streaming import StreamingGateway


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


def test_gateway_streaming():
    msg = json.dumps({"type": "trades", "symbol": "TEST", "price": 1})

    async def connect(_):
        return FakeConnection([msg])

    async def run_test():
        with patch("websockets.connect", new=connect):
            cfg = {
                "streaming": {
                    "providers": [
                        {
                            "name": "alpaca",
                            "websocket_url": "ws://test",
                            "auth_method": "none",
                            "subscriptions": ["trades"],
                        }
                    ]
                }
            }
            with tempfile.NamedTemporaryFile("w+") as f:
                yaml.safe_dump(cfg, f)
                f.flush()
                gateway = StreamingGateway(f.name)
                out = []
                gateway.subscribe_to_trades(["TEST"], callback=lambda data: out.append(data))
                await gateway._start()
                assert out == [{"type": "trades", "symbol": "TEST", "price": 1}]

    asyncio.run(run_test())
