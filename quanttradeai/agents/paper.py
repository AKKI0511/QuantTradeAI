"""Paper execution workflow for rule, LLM, and hybrid agents."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quanttradeai.data.loader import DataLoader
from quanttradeai.data.processor import DataProcessor
from quanttradeai.streaming.gateway import StreamingGateway
from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.utils.config_validator import validate_project_config
from quanttradeai.utils.project_config import (
    compile_paper_streaming_runtime_config,
    compile_research_runtime_configs,
)
from quanttradeai.utils.run_records import apply_required_run_fields, create_run_dir

from .backtest import (
    PROMPT_SAMPLE_LIMIT,
    ModelSignalRuntime,
    _agent_prompt_config,
    _build_strategy,
    _load_model_signal_sources,
)
from .base import AgentDecision, AgentSimulationState
from .context import build_context_payload

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return value
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, default=_json_default))
            handle.write("\n")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _copy_artifact(source: str | Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
    return str(destination)


def _parse_timeframe(timeframe: str) -> tuple[int, str] | None:
    normalized = str(timeframe or "").strip().lower()
    match = re.match(r"^(\d+)(mo|wk|m|h|d)$", normalized)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def _timeframe_to_pandas_freq(timeframe: str) -> str:
    parsed = _parse_timeframe(timeframe)
    if parsed is None:
        return "1D"
    amount, unit = parsed
    unit_map = {
        "m": "min",
        "h": "h",
        "d": "D",
        "wk": "W",
        "mo": "ME",
    }
    return f"{amount}{unit_map[unit]}"


def _bootstrap_window_delta(timeframe: str, bars: int) -> timedelta:
    parsed = _parse_timeframe(timeframe)
    multiplier = max(1, bars) * 3
    if parsed is None:
        return timedelta(days=multiplier)

    amount, unit = parsed
    steps = amount * multiplier
    if unit == "m":
        return timedelta(minutes=steps)
    if unit == "h":
        return timedelta(hours=steps)
    if unit == "d":
        return timedelta(days=steps)
    if unit == "wk":
        return timedelta(weeks=steps)
    return timedelta(days=steps * 30)


def _bucket_for_timestamp(timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    freq = _timeframe_to_pandas_freq(timeframe)
    if freq.endswith("W") or freq.endswith("ME"):
        bucket = timestamp.tz_convert(None) if timestamp.tzinfo else timestamp
        bucket = bucket.to_period(freq).to_timestamp()
        if timestamp.tzinfo:
            return bucket.tz_localize(timestamp.tzinfo)
        return bucket
    return timestamp.floor(freq)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df if isinstance(df.index, pd.DatetimeIndex) else df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize(timezone.utc)
    else:
        out.index = out.index.tz_convert(timezone.utc)
    return out


def _required_bootstrap_bars(agent_config: dict[str, Any]) -> int:
    market_data_cfg = dict((agent_config.get("context") or {}).get("market_data") or {})
    lookback_bars = int(market_data_cfg.get("lookback_bars", 20))
    return max(260, lookback_bars, 220)


def _build_paper_runtime_model_config(
    model_cfg: dict[str, Any],
    agent_config: dict[str, Any],
) -> dict[str, Any]:
    runtime_model_cfg = json.loads(json.dumps(model_cfg))
    data_cfg = dict(runtime_model_cfg.get("data") or {})
    now = datetime.now(timezone.utc)
    timeframe = str(data_cfg.get("timeframe") or "1d")
    recent_start = (
        now - _bootstrap_window_delta(timeframe, _required_bootstrap_bars(agent_config))
    ).date()

    start_raw = data_cfg.get("start_date")
    try:
        start_dt = datetime.fromisoformat(str(start_raw)).date()
    except (TypeError, ValueError):
        start_dt = recent_start

    data_cfg["start_date"] = min(start_dt, recent_start).isoformat()
    data_cfg["end_date"] = now.date().isoformat()
    data_cfg["test_start"] = None
    data_cfg["test_end"] = None
    runtime_model_cfg["data"] = data_cfg
    return runtime_model_cfg


def _paper_metrics(
    portfolio: PortfolioManager,
    execution_log: list[dict[str, Any]],
) -> dict[str, Any]:
    positions: dict[str, Any] = {}
    unrealized_pnl = 0.0
    for symbol, position in portfolio.positions.items():
        qty = int(position.get("qty", 0))
        current_price = _safe_float(position.get("price"), 0.0)
        entry_price = _safe_float(position.get("entry_price"), current_price)
        market_value = qty * current_price
        position_unrealized = qty * (current_price - entry_price)
        unrealized_pnl += position_unrealized
        positions[symbol] = {
            "qty": qty,
            "entry_price": entry_price,
            "current_price": current_price,
            "market_value": market_value,
            "unrealized_pnl": position_unrealized,
            "stop_loss_pct": _safe_float(position.get("stop_loss_pct"), 0.0),
        }

    portfolio_value = portfolio.portfolio_value
    return {
        "status": "available",
        "execution_count": len(execution_log),
        "realized_pnl": _safe_float(portfolio.realized_pnl, 0.0),
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": portfolio_value - portfolio.initial_capital,
        "cash": portfolio.cash,
        "portfolio_value": portfolio_value,
        "open_positions": positions,
    }


@dataclass(slots=True)
class OpenBar:
    bucket: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    last_timestamp: pd.Timestamp

    @classmethod
    def from_message(
        cls,
        *,
        bucket: pd.Timestamp,
        timestamp: pd.Timestamp,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> "OpenBar":
        return cls(
            bucket=bucket,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            last_timestamp=timestamp,
        )

    def update(
        self,
        *,
        timestamp: pd.Timestamp,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        self.high = max(self.high, high)
        self.low = min(self.low, low)
        self.close = close
        self.volume += volume
        self.last_timestamp = max(self.last_timestamp, timestamp)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Open": self.open,
                    "High": self.high,
                    "Low": self.low,
                    "Close": self.close,
                    "Volume": self.volume,
                }
            ],
            index=[self.bucket],
        )


@dataclass
class PaperAgentEngine:
    """Run a rule, LLM, or hybrid agent in paper mode from streaming bars."""

    project_config_path: str
    agent_config: dict[str, Any]
    runtime_model_config: str
    runtime_features_config: str
    runtime_streaming_config: str
    feature_definitions: list[dict[str, Any]]
    model_signal_runtimes: list[ModelSignalRuntime] = field(default_factory=list)
    gateway: StreamingGateway | None = None
    data_loader: DataLoader | None = None
    data_processor: DataProcessor | None = None
    initial_capital: float = 100_000.0
    max_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.10
    stop_loss_pct: float = 0.02
    history_window: int = 512
    shutdown_drain_timeout: float = 0.5
    prompt_sample_limit: int = PROMPT_SAMPLE_LIMIT
    _consumer: asyncio.Task | None = field(default=None, init=False)
    _history: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    _states: dict[str, AgentSimulationState] = field(default_factory=dict, init=False)
    _open_bars: dict[str, OpenBar] = field(default_factory=dict, init=False)
    decision_log: list[dict[str, Any]] = field(default_factory=list, init=False)
    execution_log: list[dict[str, Any]] = field(default_factory=list, init=False)
    prompt_samples: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.gateway = self.gateway or StreamingGateway(self.runtime_streaming_config)
        self.data_loader = self.data_loader or DataLoader(self.runtime_model_config)
        self.data_processor = self.data_processor or DataProcessor(
            self.runtime_features_config
        )
        self.strategy = _build_strategy(
            agent_config=self.agent_config,
            project_config_path=self.project_config_path,
        )
        self.include_prompt_artifacts = self.agent_config.get("kind") in {
            "llm",
            "hybrid",
        }
        self.portfolio = PortfolioManager(
            capital=self.initial_capital,
            max_risk_per_trade=self.max_risk_per_trade,
            max_portfolio_risk=self.max_portfolio_risk,
        )
        with open(self.runtime_model_config, "r", encoding="utf-8") as handle:
            runtime_model = yaml.safe_load(handle) or {}
        self.timeframe = str((runtime_model.get("data") or {}).get("timeframe") or "1d")
        self.configured_symbols = list(
            (runtime_model.get("data") or {}).get("symbols") or []
        )

    def _extract_timestamp(self, message: dict[str, Any]) -> pd.Timestamp:
        ts = (
            message.get("timestamp")
            or message.get("time")
            or message.get("t")
            or message.get("T")
            or message.get("Datetime")
            or message.get("Date")
        )
        if ts is None:
            return pd.Timestamp(datetime.now(timezone.utc))
        timestamp = pd.Timestamp(ts)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize(timezone.utc)
        return timestamp

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(message) if message else {}
        symbol = (
            message.get("symbol")
            or message.get("S")
            or message.get("sym")
            or message.get("Symbol")
            or message.get("ticker")
        )
        if symbol:
            normalized["symbol"] = symbol

        ts = (
            message.get("timestamp")
            or message.get("time")
            or message.get("t")
            or message.get("T")
            or message.get("Datetime")
            or message.get("Date")
        )
        if ts:
            normalized["timestamp"] = ts

        price = (
            message.get("price")
            or message.get("p")
            or message.get("close")
            or message.get("c")
            or message.get("ap")
            or message.get("bp")
            or message.get("mid")
            or message.get("Mid")
            or message.get("last_price")
        )
        if price is not None:
            coerced = _safe_float(price, default=0.0)
            normalized.setdefault("price", coerced)
            normalized.setdefault("close", coerced)

        open_price = message.get("open") or message.get("o")
        if open_price is not None:
            normalized.setdefault("open", open_price)

        high = message.get("high") or message.get("h")
        if high is not None:
            normalized.setdefault("high", high)

        low = message.get("low") or message.get("l")
        if low is not None:
            normalized.setdefault("low", low)

        volume = message.get("volume") or message.get("v") or message.get("V")
        if volume is not None:
            normalized.setdefault("volume", volume)

        return normalized

    def _extract_price_fields(
        self,
        message: dict[str, Any],
    ) -> tuple[float, float, float, float, float]:
        price = (
            message.get("price")
            or message.get("close")
            or message.get("Close")
            or message.get("p")
            or message.get("c")
            or message.get("mid")
            or message.get("Mid")
            or 0.0
        )
        open_price = message.get("open") or message.get("Open") or message.get("o")
        high = message.get("high") or message.get("High") or message.get("h")
        low = message.get("low") or message.get("Low") or message.get("l")
        close = (
            message.get("close") or message.get("Close") or message.get("c") or price
        )
        volume = (
            message.get("volume")
            or message.get("Volume")
            or message.get("v")
            or message.get("V")
            or 0.0
        )
        fallback_price = _safe_float(price, default=0.0)
        open_f = _safe_float(open_price, default=fallback_price)
        high_f = _safe_float(high, default=fallback_price)
        low_f = _safe_float(low, default=fallback_price)
        close_f = _safe_float(close, default=fallback_price)
        volume_f = _safe_float(volume, default=0.0)
        return open_f, high_f, low_f, close_f, volume_f

    def _mark_to_market(self, symbol: str, price: float) -> None:
        if symbol in self.portfolio.positions:
            self.portfolio.positions[symbol]["price"] = price

    def _append_history(self, symbol: str, frame: pd.DataFrame) -> None:
        history = self._history.get(symbol)
        history = pd.concat([history, frame]) if history is not None else frame
        history = history.sort_index()
        history = history[~history.index.duplicated(keep="last")]
        self._history[symbol] = history.tail(self.history_window)

    def bootstrap_history(self) -> None:
        bootstrap_frames = self.data_loader.fetch_data()
        now_bucket = _bucket_for_timestamp(
            pd.Timestamp(datetime.now(timezone.utc)),
            self.timeframe,
        )
        for symbol, frame in bootstrap_frames.items():
            history = _ensure_datetime_index(frame).sort_index()
            history = history[
                [
                    column
                    for column in ("Open", "High", "Low", "Close", "Volume")
                    if column in history.columns
                ]
            ].copy()
            if history.empty:
                continue

            latest_bucket = _bucket_for_timestamp(
                pd.Timestamp(history.index.max()),
                self.timeframe,
            )
            if latest_bucket == now_bucket and len(history) >= 1:
                latest = history.iloc[-1]
                self._open_bars[symbol] = OpenBar.from_message(
                    bucket=latest_bucket,
                    timestamp=pd.Timestamp(history.index.max()),
                    open_price=_safe_float(latest.get("Open"), 0.0),
                    high=_safe_float(latest.get("High"), 0.0),
                    low=_safe_float(latest.get("Low"), 0.0),
                    close=_safe_float(latest.get("Close"), 0.0),
                    volume=_safe_float(latest.get("Volume"), 0.0),
                )
                history = history.iloc[:-1]

            if history.empty:
                continue
            self._history[symbol] = history.tail(self.history_window)

    def _update_open_bar(
        self,
        *,
        symbol: str,
        timestamp: pd.Timestamp,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> tuple[pd.Timestamp, pd.DataFrame] | None:
        bucket = _bucket_for_timestamp(timestamp, self.timeframe)
        current_bar = self._open_bars.get(symbol)

        if current_bar is None:
            latest_history = self._history.get(symbol)
            if latest_history is not None and not latest_history.empty:
                latest_bucket = _bucket_for_timestamp(
                    pd.Timestamp(latest_history.index.max()),
                    self.timeframe,
                )
                if bucket <= latest_bucket:
                    return None

            self._open_bars[symbol] = OpenBar.from_message(
                bucket=bucket,
                timestamp=timestamp,
                open_price=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
            return None

        if bucket < current_bar.bucket:
            return None

        if bucket == current_bar.bucket:
            current_bar.update(
                timestamp=timestamp,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
            return None

        completed = current_bar.to_frame()
        completed_timestamp = current_bar.bucket
        self._open_bars[symbol] = OpenBar.from_message(
            bucket=bucket,
            timestamp=timestamp,
            open_price=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        return completed_timestamp, completed

    def _apply_decision(
        self,
        *,
        symbol: str,
        timestamp: pd.Timestamp,
        price: float,
        decision: AgentDecision,
    ) -> dict[str, Any]:
        had_position = symbol in self.portfolio.positions
        position_before = 1 if had_position else 0
        desired_target = (
            1
            if decision.action == "buy"
            else -1 if decision.action == "sell" else position_before
        )
        execution_status = "hold"
        execution_payload: dict[str, Any] | None = None

        if decision.action == "buy":
            if had_position:
                execution_status = "already_long"
            else:
                qty = self.portfolio.open_position(
                    symbol,
                    price,
                    stop_loss_pct=self.stop_loss_pct,
                )
                if qty > 0:
                    execution_status = "executed"
                    execution_payload = {
                        "action": "buy",
                        "symbol": symbol,
                        "qty": qty,
                        "price": price,
                        "timestamp": timestamp,
                        "decision_action": decision.action,
                    }
                    self.execution_log.append(execution_payload)
                else:
                    execution_status = "blocked"
        elif decision.action == "sell":
            if not had_position:
                execution_status = "no_position"
            else:
                qty = self.portfolio.close_position(symbol, price)
                if qty > 0:
                    execution_status = "executed"
                    execution_payload = {
                        "action": "sell",
                        "symbol": symbol,
                        "qty": qty,
                        "price": price,
                        "timestamp": timestamp,
                        "decision_action": decision.action,
                    }
                    self.execution_log.append(execution_payload)
                else:
                    execution_status = "blocked"

        self._mark_to_market(symbol, price)
        position_after = 1 if symbol in self.portfolio.positions else 0
        state = self._states.setdefault(symbol, AgentSimulationState())
        state.target_position = position_after
        state.last_action = decision.action
        state.last_reason = decision.reason
        state.decision_count += 1

        return {
            "desired_target_position": desired_target,
            "target_position": position_after,
            "position_before": position_before,
            "position_after": position_after,
            "execution_status": execution_status,
            "execution": execution_payload,
        }

    def _process_completed_bar(
        self,
        *,
        symbol: str,
        completed_timestamp: pd.Timestamp,
    ) -> None:
        history = self._history.get(symbol)
        if history is None or history.empty:
            return

        featured_history = self.data_processor.generate_features(history.copy())
        if (
            featured_history.empty
            or featured_history.index.max() != completed_timestamp
        ):
            return

        current_frame = featured_history.tail(1)
        current_row = current_frame.iloc[0]
        state = self._states.setdefault(symbol, AgentSimulationState())
        runtime_agent_config = _agent_prompt_config(self.agent_config, symbol)
        model_signals = {
            runtime.name: runtime.predict(current_frame)
            for runtime in self.model_signal_runtimes
        }
        context = build_context_payload(
            feature_definitions=self.feature_definitions,
            agent_config=runtime_agent_config,
            history=featured_history,
            current_row=current_row,
            model_signals=model_signals,
            state=state,
        )
        decision = self.strategy.decide(
            agent_name=str(self.agent_config.get("name") or ""),
            symbol=symbol,
            timestamp=completed_timestamp,
            context=context,
            tools=list(self.agent_config.get("tools") or []),
        )
        price = _safe_float(history["Close"].iloc[-1], 0.0)
        execution_result = self._apply_decision(
            symbol=symbol,
            timestamp=completed_timestamp,
            price=price,
            decision=decision,
        )
        decision_record = {
            "symbol": symbol,
            "timestamp": completed_timestamp,
            "action": decision.action,
            "reason": decision.reason,
            "context": context,
            "model_signals": {
                name: {"signal": signal} for name, signal in model_signals.items()
            },
            **execution_result,
        }
        self.decision_log.append(decision_record)
        if (
            self.include_prompt_artifacts
            and len(self.prompt_samples) < self.prompt_sample_limit
        ):
            self.prompt_samples.append(
                {
                    "symbol": symbol,
                    "timestamp": completed_timestamp,
                    "prompt_payload": decision.prompt_payload,
                    "response_payload": decision.response_payload,
                    "raw_response": decision.raw_response,
                }
            )

    async def _consume_buffer(self) -> None:
        while True:
            message = await self.gateway.buffer.get()
            try:
                normalized = self._normalize_message(message)
                symbol = normalized.get("symbol")
                if not symbol:
                    continue
                timestamp = self._extract_timestamp(normalized)
                open_price, high, low, close, volume = self._extract_price_fields(
                    normalized
                )
                if close <= 0:
                    continue

                self._mark_to_market(symbol, close)
                completed = self._update_open_bar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open_price=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
                if completed is None:
                    continue

                completed_timestamp, frame = completed
                self._append_history(symbol, frame)
                self._mark_to_market(
                    symbol,
                    _safe_float(frame["Close"].iloc[-1], close),
                )
                self._process_completed_bar(
                    symbol=symbol,
                    completed_timestamp=completed_timestamp,
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.error("paper_agent_error", exc_info=exc)

    async def start(self) -> None:
        self.bootstrap_history()
        self._consumer = asyncio.create_task(self._consume_buffer())
        try:
            await self.gateway._start()
        finally:
            if self._consumer is not None:
                if self.shutdown_drain_timeout > 0:
                    deadline = (
                        asyncio.get_running_loop().time() + self.shutdown_drain_timeout
                    )
                    while (
                        not self.gateway.buffer.queue.empty()
                        and asyncio.get_running_loop().time() < deadline
                    ):
                        await asyncio.sleep(0.01)
                self._consumer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._consumer


def run_agent_paper(
    *,
    project_config_path: str = "config/project.yaml",
    agent_name: str,
) -> dict[str, Any]:
    """Run a rule, LLM, or hybrid agent in paper mode using streaming input."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir, run_id = create_run_dir(
        run_type="agent",
        mode="paper",
        name=agent_name,
        timestamp=timestamp,
    )
    summary: dict[str, Any] = {
        "agent_name": agent_name,
        "status": "failed",
        "timestamps": {"started_at": datetime.now(timezone.utc).isoformat()},
        "artifacts": {},
        "warnings": [],
        "symbols": [],
    }
    apply_required_run_fields(
        summary,
        run_dir=run_dir,
        run_type="agent",
        mode="paper",
        name=agent_name,
    )
    summary["run_id"] = run_id

    try:
        validation = validate_project_config(
            config_path=project_config_path,
            output_dir=run_dir / "validation",
            timestamp_subdir=False,
        )
        summary["warnings"] = list(validation.get("warnings", []))
        resolved_path = Path(validation["artifacts"]["resolved_config"])
        root_resolved_path = run_dir / "resolved_project_config.yaml"
        summary["artifacts"]["resolved_project_config"] = _copy_artifact(
            resolved_path,
            root_resolved_path,
        )
        summary["artifacts"]["validation_summary"] = validation["artifacts"]["summary"]
        project_config = (
            yaml.safe_load(root_resolved_path.read_text(encoding="utf-8")) or {}
        )

        agent_config = next(
            (
                dict(agent)
                for agent in project_config.get("agents") or []
                if agent.get("name") == agent_name
            ),
            None,
        )
        if agent_config is None:
            raise ValueError(f"Agent '{agent_name}' not found in project config.")
        if agent_config.get("kind") not in {"rule", "llm", "hybrid"}:
            raise ValueError(
                f"Agent '{agent_name}' is kind={agent_config.get('kind')}; expected kind=rule, kind=llm, or kind=hybrid."
            )

        model_cfg, features_cfg, _ = compile_research_runtime_configs(
            project_config,
            require_research=False,
        )
        runtime_model_cfg = _build_paper_runtime_model_config(model_cfg, agent_config)
        runtime_model_path = run_dir / "runtime_model_config.yaml"
        runtime_features_path = run_dir / "runtime_features_config.yaml"
        runtime_streaming_path = run_dir / "runtime_streaming_config.yaml"
        runtime_model_path.write_text(
            yaml.safe_dump(runtime_model_cfg, sort_keys=False),
            encoding="utf-8",
        )
        runtime_features_path.write_text(
            yaml.safe_dump(features_cfg, sort_keys=False),
            encoding="utf-8",
        )
        runtime_streaming_cfg = compile_paper_streaming_runtime_config(project_config)
        runtime_streaming_path.write_text(
            yaml.safe_dump(runtime_streaming_cfg, sort_keys=False),
            encoding="utf-8",
        )
        summary["artifacts"]["runtime_model_config"] = str(runtime_model_path)
        summary["artifacts"]["runtime_features_config"] = str(runtime_features_path)
        summary["artifacts"]["runtime_streaming_config"] = str(runtime_streaming_path)

        feature_definitions = list(
            (project_config.get("features") or {}).get("definitions") or []
        )
        model_signal_runtimes = _load_model_signal_sources(
            agent_config=agent_config,
            project_config_path=project_config_path,
            runtime_model_config_path=runtime_model_path,
        )
        trading_cfg = dict(runtime_model_cfg.get("trading") or {})
        agent_risk_cfg = dict(agent_config.get("risk") or {})
        stop_loss_pct = _safe_float(trading_cfg.get("stop_loss"), 0.02)
        initial_capital = _safe_float(trading_cfg.get("initial_capital"), 100_000.0)
        max_risk_per_trade = _safe_float(
            agent_risk_cfg.get("max_position_pct"),
            _safe_float(trading_cfg.get("max_risk_per_trade"), 0.02),
        )
        max_portfolio_risk = _safe_float(
            agent_risk_cfg.get("max_portfolio_risk"),
            _safe_float(trading_cfg.get("max_portfolio_risk"), max_risk_per_trade),
        )
        history_window = max(
            _required_bootstrap_bars(agent_config) + 32,
            int(
                ((agent_config.get("context") or {}).get("market_data") or {}).get(
                    "lookback_bars",
                    20,
                )
            )
            + 32,
        )

        engine = PaperAgentEngine(
            project_config_path=project_config_path,
            agent_config=agent_config,
            runtime_model_config=str(runtime_model_path),
            runtime_features_config=str(runtime_features_path),
            runtime_streaming_config=str(runtime_streaming_path),
            feature_definitions=feature_definitions,
            model_signal_runtimes=model_signal_runtimes,
            initial_capital=initial_capital,
            max_risk_per_trade=max_risk_per_trade,
            max_portfolio_risk=max_portfolio_risk,
            stop_loss_pct=stop_loss_pct,
            history_window=history_window,
        )
        asyncio.run(engine.start())

        _write_jsonl(run_dir / "decisions.jsonl", engine.decision_log)
        _write_jsonl(run_dir / "executions.jsonl", engine.execution_log)
        metrics_payload = _paper_metrics(engine.portfolio, engine.execution_log)
        metrics_payload["decision_count"] = len(engine.decision_log)
        _write_json(run_dir / "metrics.json", metrics_payload)

        artifacts = {
            **summary["artifacts"],
            "metrics": str(run_dir / "metrics.json"),
            "decisions": str(run_dir / "decisions.jsonl"),
            "executions": str(run_dir / "executions.jsonl"),
        }
        if engine.include_prompt_artifacts:
            _write_json(run_dir / "prompt_samples.json", engine.prompt_samples)
            artifacts["prompt_samples"] = str(run_dir / "prompt_samples.json")
        summary.update(
            {
                "status": "success",
                "agent_kind": agent_config.get("kind"),
                "symbols": sorted(
                    set(engine.configured_symbols)
                    | set(engine._history.keys())
                    | set(engine._open_bars.keys())
                ),
                "decision_count": len(engine.decision_log),
                "execution_count": len(engine.execution_log),
                "artifacts": artifacts,
                "metrics": metrics_payload,
            }
        )
        summary["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        apply_required_run_fields(
            summary,
            run_dir=run_dir,
            run_type="agent",
            mode="paper",
            name=agent_name,
        )
        _write_json(run_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        logger.error("paper_agent_run_failed", exc_info=exc)
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["timestamps"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        apply_required_run_fields(
            summary,
            run_dir=run_dir,
            run_type="agent",
            mode="paper",
            name=agent_name,
        )
        _write_json(run_dir / "summary.json", summary)
        raise
