"""Alpaca REST broker implementation for paper and live execution."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib import error, request

from .base import (
    BrokerAccountSnapshot,
    BrokerClient,
    BrokerCredentialsError,
    BrokerError,
    BrokerOrderResult,
    BrokerPositionSnapshot,
    _safe_float,
    _safe_int,
)

ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE_URL = "https://api.alpaca.markets"
ALPACA_TERMINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "expired",
    "rejected",
    "done_for_day",
    "stopped",
    "suspended",
    "calculated",
}


def _parse_timestamp(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


class AlpacaBrokerClient(BrokerClient):
    """Happy-path Alpaca trading client using REST polling only."""

    provider = "alpaca"

    def __init__(
        self,
        *,
        mode: str = "paper",
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        request_timeout: float = 10.0,
        poll_interval: float = 0.5,
        order_timeout: float = 15.0,
    ) -> None:
        self.mode = str(mode or "paper").strip().lower()
        self.api_key = str(api_key or os.environ.get("ALPACA_API_KEY") or "").strip()
        self.api_secret = str(
            api_secret or os.environ.get("ALPACA_API_SECRET") or ""
        ).strip()
        if not self.api_key or not self.api_secret:
            raise BrokerCredentialsError(
                "ALPACA_API_KEY and ALPACA_API_SECRET are required for Alpaca-backed execution."
            )

        if base_url:
            self.base_url = base_url.rstrip("/")
        elif self.mode == "live":
            self.base_url = ALPACA_LIVE_BASE_URL
        else:
            self.base_url = ALPACA_PAPER_BASE_URL

        self.request_timeout = float(request_timeout)
        self.poll_interval = float(poll_interval)
        self.order_timeout = float(order_timeout)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Accept": "application/json",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        req = request.Request(url, method=method.upper(), data=data, headers=headers)
        try:
            with request.urlopen(req, timeout=self.request_timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise BrokerError(
                f"Alpaca request failed: {method.upper()} {path} returned {exc.code}: {details or exc.reason}"
            ) from exc
        except error.URLError as exc:
            raise BrokerError(f"Alpaca request failed: {exc.reason}") from exc

        if not body.strip():
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise BrokerError(
                f"Alpaca returned invalid JSON for {method.upper()} {path}."
            ) from exc

    def _parse_account(self, payload: dict[str, Any]) -> BrokerAccountSnapshot:
        return BrokerAccountSnapshot(
            account_id=str(payload.get("id") or ""),
            cash=_safe_float(payload.get("cash"), 0.0),
            equity=_safe_float(payload.get("equity"), 0.0),
            buying_power=_safe_float(payload.get("buying_power"), 0.0),
            currency=str(payload.get("currency") or "USD"),
            status=str(payload.get("status") or ""),
            raw=dict(payload),
        )

    def _parse_position(self, payload: dict[str, Any]) -> BrokerPositionSnapshot:
        return BrokerPositionSnapshot(
            symbol=str(payload.get("symbol") or ""),
            qty=abs(_safe_int(payload.get("qty"), 0)),
            market_price=_safe_float(payload.get("current_price"), 0.0),
            avg_entry_price=_safe_float(payload.get("avg_entry_price"), 0.0),
            side=str(payload.get("side") or "long"),
            unrealized_pnl=_safe_float(payload.get("unrealized_pl"), 0.0),
            raw=dict(payload),
        )

    def _parse_order(self, payload: dict[str, Any]) -> BrokerOrderResult:
        return BrokerOrderResult(
            order_id=str(payload.get("id") or ""),
            symbol=str(payload.get("symbol") or ""),
            action=str(payload.get("side") or ""),
            qty=abs(_safe_int(payload.get("qty"), 0)),
            status=str(payload.get("status") or ""),
            submitted_at=_parse_timestamp(payload.get("submitted_at")),
            filled_at=_parse_timestamp(payload.get("filled_at")),
            filled_qty=abs(_safe_int(payload.get("filled_qty"), 0)),
            filled_avg_price=(
                _safe_float(payload.get("filled_avg_price"), 0.0)
                if payload.get("filled_avg_price") not in (None, "")
                else None
            ),
            raw=dict(payload),
        )

    def get_account(self) -> BrokerAccountSnapshot:
        payload = self._request("GET", "/v2/account")
        return self._parse_account(dict(payload or {}))

    def list_positions(self) -> list[BrokerPositionSnapshot]:
        payload = self._request("GET", "/v2/positions")
        return [
            self._parse_position(dict(item or {}))
            for item in list(payload or [])
            if dict(item or {}).get("symbol")
        ]

    def submit_market_order(
        self,
        *,
        symbol: str,
        action: str,
        qty: int,
    ) -> BrokerOrderResult:
        payload = self._request(
            "POST",
            "/v2/orders",
            payload={
                "symbol": symbol,
                "qty": str(int(qty)),
                "side": str(action).strip().lower(),
                "type": "market",
                "time_in_force": "day",
            },
        )
        return self._parse_order(dict(payload or {}))

    def get_order(self, order_id: str) -> BrokerOrderResult:
        payload = self._request("GET", f"/v2/orders/{order_id}")
        return self._parse_order(dict(payload or {}))

    def wait_for_order(
        self,
        order_id: str,
        *,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> BrokerOrderResult:
        interval = self.poll_interval if poll_interval is None else float(poll_interval)
        deadline = time.monotonic() + (
            self.order_timeout if timeout is None else float(timeout)
        )
        latest = self.get_order(order_id)
        while latest.status not in ALPACA_TERMINAL_ORDER_STATUSES:
            if time.monotonic() >= deadline:
                return latest
            time.sleep(max(interval, 0.0))
            latest = self.get_order(order_id)
        return latest
