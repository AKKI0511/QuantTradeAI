import json

import pytest

from quanttradeai.brokers.alpaca import AlpacaBrokerClient
from quanttradeai.brokers.base import BrokerOrderResult


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


def test_alpaca_get_account_and_positions_parse_payloads(monkeypatch):
    responses = iter(
        [
            {
                "id": "acct-123",
                "cash": "1000.50",
                "equity": "1020.25",
                "buying_power": "2500.00",
                "currency": "USD",
                "status": "ACTIVE",
            },
            [
                {
                    "symbol": "AAPL",
                    "qty": "3",
                    "side": "long",
                    "current_price": "201.5",
                    "avg_entry_price": "199.0",
                    "unrealized_pl": "7.5",
                }
            ],
        ]
    )

    def _fake_urlopen(req, timeout):
        return _FakeHTTPResponse(next(responses))

    monkeypatch.setattr("quanttradeai.brokers.alpaca.request.urlopen", _fake_urlopen)
    client = AlpacaBrokerClient(api_key="key", api_secret="secret")

    account = client.get_account()
    positions = client.list_positions()

    assert account.account_id == "acct-123"
    assert account.cash == pytest.approx(1000.50)
    assert account.equity == pytest.approx(1020.25)
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 3
    assert positions[0].market_price == pytest.approx(201.5)
    assert positions[0].avg_entry_price == pytest.approx(199.0)


def test_alpaca_submit_market_order_builds_expected_payload(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "id": "ord-1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "5",
                "status": "new",
                "filled_qty": "0",
            }
        )

    monkeypatch.setattr("quanttradeai.brokers.alpaca.request.urlopen", _fake_urlopen)
    client = AlpacaBrokerClient(api_key="key", api_secret="secret", mode="paper")

    order = client.submit_market_order(symbol="AAPL", action="buy", qty=5)

    assert captured["url"].endswith("/v2/orders")
    assert captured["method"] == "POST"
    assert captured["payload"] == {
        "symbol": "AAPL",
        "qty": "5",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }
    assert order.order_id == "ord-1"
    assert order.status == "new"


def test_alpaca_wait_for_order_polls_until_filled(monkeypatch):
    client = AlpacaBrokerClient(
        api_key="key",
        api_secret="secret",
        poll_interval=0.0,
        order_timeout=1.0,
    )
    responses = iter(
        [
            BrokerOrderResult(
                order_id="ord-1",
                symbol="AAPL",
                action="buy",
                qty=5,
                status="new",
            ),
            BrokerOrderResult(
                order_id="ord-1",
                symbol="AAPL",
                action="buy",
                qty=5,
                status="partially_filled",
                filled_qty=2,
                filled_avg_price=101.0,
            ),
            BrokerOrderResult(
                order_id="ord-1",
                symbol="AAPL",
                action="buy",
                qty=5,
                status="filled",
                filled_qty=5,
                filled_avg_price=101.5,
            ),
        ]
    )
    monkeypatch.setattr(client, "get_order", lambda order_id: next(responses))

    result = client.wait_for_order("ord-1", poll_interval=0.0, timeout=0.1)

    assert result.status == "filled"
    assert result.filled_qty == 5
    assert result.filled_avg_price == pytest.approx(101.5)


@pytest.mark.parametrize("status", ["canceled", "rejected"])
def test_alpaca_wait_for_order_returns_terminal_failure_status(monkeypatch, status):
    client = AlpacaBrokerClient(
        api_key="key",
        api_secret="secret",
        poll_interval=0.0,
        order_timeout=1.0,
    )
    monkeypatch.setattr(
        client,
        "get_order",
        lambda order_id: BrokerOrderResult(
            order_id=order_id,
            symbol="AAPL",
            action="sell",
            qty=5,
            status=status,
        ),
    )

    result = client.wait_for_order("ord-2", poll_interval=0.0, timeout=0.1)

    assert result.status == status
