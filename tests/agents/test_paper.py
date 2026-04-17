import asyncio
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from quanttradeai.agents.paper import (
    PaperAgentEngine,
    _build_paper_runtime_model_config,
    _required_bootstrap_bars,
)
from quanttradeai.streaming.replay import ReplayGateway
from quanttradeai.streaming.stream_buffer import StreamBuffer


def _mock_history(periods: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "Open": np.linspace(100.0, 120.0, len(index)),
            "High": np.linspace(101.0, 121.0, len(index)),
            "Low": np.linspace(99.0, 119.0, len(index)),
            "Close": np.linspace(100.5, 120.5, len(index)),
            "Volume": np.linspace(1000.0, 1400.0, len(index)),
        },
        index=index,
    )


class FakeLoader:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames

    def fetch_data(self):
        return self.frames


class DummyProcessor:
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        featured = df.copy()
        featured["rsi"] = np.linspace(45.0, 65.0, len(featured))
        featured["volume_momentum_20"] = np.linspace(0.1, 0.3, len(featured))
        return featured


class DropTextSensitiveProcessor:
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        featured = df.copy()
        if "text" in featured.columns:
            featured = featured.dropna()
        featured["rsi"] = np.linspace(45.0, 65.0, len(featured))
        return featured


class ScriptedGateway:
    def __init__(self, messages: list[dict]) -> None:
        self.buffer = StreamBuffer(32)
        self.messages = messages

    async def _start(self) -> None:
        for message in self.messages:
            await self.buffer.put(message)


class FakeSignalRuntime:
    def __init__(self, name: str, signal: int) -> None:
        self.name = name
        self.signal = signal

    def predict(self, features_frame: pd.DataFrame) -> int:
        return self.signal


def _completion_from_actions(actions: list[str]):
    action_iter = iter(actions)

    def _fake_completion(**kwargs):
        action = next(action_iter, "hold")
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"action": action, "reason": f"{action} signal"}
                        )
                    }
                }
            ]
        }

    return _fake_completion


def _build_engine(
    tmp_path: Path,
    monkeypatch,
    *,
    kind: str = "llm",
    gateway_messages: list[dict],
    completion,
    model_signal_runtimes=None,
    context_overrides: dict | None = None,
    history_frame: pd.DataFrame | None = None,
    notes_content: str | None = None,
    processor=None,
) -> PaperAgentEngine:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("quanttradeai.agents.llm.completion", completion)

    project_path = tmp_path / "config" / "project.yaml"
    prompt_path = tmp_path / "prompts" / "agent.md"
    runtime_model_path = tmp_path / "runtime_model.yaml"
    project_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("Return JSON only.", encoding="utf-8")
    project_path.write_text("project:\n  name: test\n", encoding="utf-8")
    runtime_model_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "symbols": ["AAPL"],
                    "timeframe": "1d",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
                "trading": {
                    "initial_capital": 100000,
                    "stop_loss": 0.02,
                    "max_risk_per_trade": 0.02,
                    "max_portfolio_risk": 0.10,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    agent_config = {
        "name": "paper_agent",
        "kind": kind,
        "mode": "paper",
        "llm": {
            "provider": "openai",
            "model": "gpt-5.3",
            "prompt_file": "prompts/agent.md",
        },
        "context": {
            "market_data": {"enabled": True, "timeframe": "1d", "lookback_bars": 2},
            "features": ["rsi_14"],
            "model_signals": ["trend_model"] if kind == "hybrid" else [],
            "positions": True,
            "risk_state": True,
        },
        "tools": ["get_quote", "place_order"],
        "risk": {"max_position_pct": 0.05},
    }
    if context_overrides:
        agent_config["context"].update(context_overrides)

    if agent_config["context"].get("notes"):
        notes_path = tmp_path / "notes" / "paper_agent.md"
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(
            notes_content or "Trade only when the signal is clear.",
            encoding="utf-8",
        )

    return PaperAgentEngine(
        project_config_path=str(project_path),
        agent_config=agent_config,
        runtime_model_config=str(runtime_model_path),
        runtime_features_config=str(tmp_path / "runtime_features.yaml"),
        runtime_streaming_config=str(tmp_path / "runtime_streaming.yaml"),
        feature_definitions=[
            {"name": "rsi_14", "type": "technical", "params": {"period": 14}}
        ],
        model_signal_runtimes=model_signal_runtimes or [],
        gateway=ScriptedGateway(gateway_messages),
        data_loader=FakeLoader(
            {"AAPL": history_frame if history_frame is not None else _mock_history()}
        ),
        data_processor=processor or DummyProcessor(),
        history_window=512,
    )


def test_paper_engine_warm_starts_history_from_loader(tmp_path: Path, monkeypatch):
    engine = _build_engine(
        tmp_path,
        monkeypatch,
        gateway_messages=[],
        completion=_completion_from_actions([]),
    )

    engine.bootstrap_history()

    assert "AAPL" in engine._history
    assert len(engine._history["AAPL"]) == 260


def test_runtime_model_config_extends_start_date_for_bootstrap_window():
    model_cfg = {
        "data": {
            "symbols": ["AAPL"],
            "timeframe": "1d",
            "start_date": "2100-01-01",
            "end_date": "2100-12-31",
        }
    }
    agent_config = {"context": {"market_data": {"lookback_bars": 20}}}

    runtime_cfg = _build_paper_runtime_model_config(model_cfg, agent_config)

    assert runtime_cfg["data"]["start_date"] < model_cfg["data"]["start_date"]
    assert runtime_cfg["data"]["test_start"] is None
    assert runtime_cfg["data"]["test_end"] is None
    assert runtime_cfg["data"]["end_date"] <= pd.Timestamp.now(tz="UTC").date().isoformat()
    assert _required_bootstrap_bars(agent_config) == 260


def test_paper_engine_aggregates_messages_once_per_completed_bar(
    tmp_path: Path, monkeypatch
):
    engine = _build_engine(
        tmp_path,
        monkeypatch,
        gateway_messages=[
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-09-17T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 122.0,
                "volume": 15,
                "timestamp": "2024-09-17T12:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 123.0,
                "volume": 20,
                "timestamp": "2024-09-18T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 124.0,
                "volume": 25,
                "timestamp": "2024-09-19T00:00:00Z",
            },
        ],
        completion=_completion_from_actions(["buy", "hold"]),
    )

    asyncio.run(engine.start())

    assert len(engine.decision_log) == 2
    assert len(engine.execution_log) == 1
    assert engine.execution_log[0]["action"] == "buy"
    assert engine.decision_log[0]["timestamp"] == pd.Timestamp("2024-09-17T00:00:00Z")
    assert engine.decision_log[1]["timestamp"] == pd.Timestamp("2024-09-18T00:00:00Z")


def test_paper_engine_sell_without_position_is_audited_as_no_op(
    tmp_path: Path, monkeypatch
):
    engine = _build_engine(
        tmp_path,
        monkeypatch,
        gateway_messages=[
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-09-17T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 120.0,
                "volume": 10,
                "timestamp": "2024-09-18T00:00:00Z",
            },
        ],
        completion=_completion_from_actions(["sell"]),
    )

    asyncio.run(engine.start())

    assert len(engine.decision_log) == 1
    assert engine.execution_log == []
    assert engine.decision_log[0]["execution_status"] == "no_position"
    assert engine.decision_log[0]["desired_target_position"] == -1
    assert engine.decision_log[0]["target_position"] == 0


def test_paper_engine_records_prompt_context_and_hybrid_model_signals(
    tmp_path: Path, monkeypatch
):
    engine = _build_engine(
        tmp_path,
        monkeypatch,
        kind="hybrid",
        gateway_messages=[
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-09-17T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 123.0,
                "volume": 12,
                "timestamp": "2024-09-18T00:00:00Z",
            },
        ],
        completion=_completion_from_actions(["buy"]),
        model_signal_runtimes=[FakeSignalRuntime("trend_model", 1)],
    )

    asyncio.run(engine.start())

    assert len(engine.decision_log) == 1
    context = engine.decision_log[0]["context"]
    assert context["features"]["rsi_14"]["rsi"] > 0
    assert context["positions"]["target_position"] == 0
    assert context["risk_state"]["risk_limits"]["max_position_pct"] == 0.05
    assert context["model_signals"]["trend_model"]["signal"] == 1
    assert len(context["market_data"]["bars"]) == 2
    assert engine.prompt_samples
    assert "messages" in engine.prompt_samples[0]["prompt_payload"]


def test_paper_engine_context_blocks_include_orders_memory_news_and_notes(
    tmp_path: Path, monkeypatch
):
    history = _mock_history()
    history["text"] = ""
    history.iloc[-3, history.columns.get_loc("text")] = "Analyst upgrades Apple"
    history.iloc[-2, history.columns.get_loc("text")] = "Apple expands buyback"
    history.iloc[-1, history.columns.get_loc("text")] = "Apple expands buyback"

    engine = _build_engine(
        tmp_path,
        monkeypatch,
        gateway_messages=[
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-09-17T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 123.0,
                "volume": 12,
                "timestamp": "2024-09-18T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 124.0,
                "volume": 14,
                "timestamp": "2024-09-19T00:00:00Z",
            },
        ],
        completion=_completion_from_actions(["buy", "hold"]),
        history_frame=history,
        context_overrides={
            "orders": {"enabled": True, "max_entries": 2},
            "memory": {"enabled": True, "max_entries": 2},
            "news": {"enabled": True, "max_items": 2},
            "notes": True,
        },
        notes_content="Prefer continuation trades over counter-trend calls.",
    )

    asyncio.run(engine.start())

    assert len(engine.decision_log) == 2
    first_context = engine.decision_log[0]["context"]
    assert first_context["orders"] == {"recent_orders": []}
    assert first_context["memory"] == {"recent_decisions": []}
    assert first_context["news"]["headlines"] == [
        {
            "timestamp": "2024-09-16T00:00:00+00:00",
            "text": "Apple expands buyback",
        },
        {
            "timestamp": "2024-09-14T00:00:00+00:00",
            "text": "Analyst upgrades Apple",
        },
    ]
    assert first_context["notes"] == {
        "path": "notes/paper_agent.md",
        "content": "Prefer continuation trades over counter-trend calls.",
    }

    second_context = engine.decision_log[1]["context"]
    assert second_context["orders"]["recent_orders"] == [
        {
            "timestamp": "2024-09-17T00:00:00+00:00",
            "action": "buy",
            "qty": engine.execution_log[0]["qty"],
            "price": 121.0,
            "status": "executed",
        }
    ]
    assert second_context["memory"]["recent_decisions"] == [
        {
            "timestamp": "2024-09-17T00:00:00+00:00",
            "action": "buy",
            "reason": "buy signal",
            "execution_status": "executed",
            "target_position_after": 1,
        }
    ]


def test_paper_engine_sparse_news_does_not_drop_completed_bar_decisions(
    tmp_path: Path, monkeypatch
):
    history = _mock_history()
    history["text"] = pd.Series([np.nan] * len(history), index=history.index, dtype=object)
    history.iloc[-2, history.columns.get_loc("text")] = "Apple updates guidance"

    engine = _build_engine(
        tmp_path,
        monkeypatch,
        gateway_messages=[
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-09-17T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 123.0,
                "volume": 12,
                "timestamp": "2024-09-18T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 124.0,
                "volume": 14,
                "timestamp": "2024-09-19T00:00:00Z",
            },
        ],
        completion=_completion_from_actions(["buy", "hold"]),
        history_frame=history,
        context_overrides={"news": {"enabled": True, "max_items": 1}},
        processor=DropTextSensitiveProcessor(),
    )

    asyncio.run(engine.start())

    assert len(engine.decision_log) == 2
    assert engine.decision_log[0]["context"]["news"]["headlines"] == [
        {
            "timestamp": "2024-09-15T00:00:00+00:00",
            "text": "Apple updates guidance",
        }
    ]


def test_paper_engine_replay_bootstrap_does_not_double_count_history(
    tmp_path: Path, monkeypatch
):
    engine = _build_engine(
        tmp_path,
        monkeypatch,
        gateway_messages=[],
        completion=_completion_from_actions(["buy", "hold"]),
    )
    bootstrap_history = {"AAPL": _mock_history(periods=2)}
    replay_frames = {"AAPL": _mock_history(periods=4).iloc[2:4]}

    engine.bootstrap_history_frames = bootstrap_history
    engine.gateway = ReplayGateway(replay_frames, pace_delay_ms=0, buffer_size=16)

    asyncio.run(engine.start())

    assert len(engine.decision_log) == 2
    assert [entry["timestamp"] for entry in engine.decision_log] == [
        pd.Timestamp("2024-01-03T00:00:00Z"),
        pd.Timestamp("2024-01-04T00:00:00Z"),
    ]
    assert len(engine.execution_log) == 1
    assert engine.execution_log[0]["action"] == "buy"
    assert len(engine._history["AAPL"]) == 4


def test_paper_engine_replay_waits_for_completion_signal(
    tmp_path: Path, monkeypatch
):
    def _slow_completion(**kwargs):
        time.sleep(0.12)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"action": "hold", "reason": "slow replay test"}
                        )
                    }
                }
            ]
        }

    engine = _build_engine(
        tmp_path,
        monkeypatch,
        gateway_messages=[],
        completion=_slow_completion,
    )
    bootstrap_history = {"AAPL": _mock_history(periods=2)}
    replay_frames = {"AAPL": _mock_history(periods=8).iloc[2:8]}

    engine.bootstrap_history_frames = bootstrap_history
    engine.gateway = ReplayGateway(replay_frames, pace_delay_ms=0, buffer_size=32)

    asyncio.run(engine.start())

    assert len(engine.decision_log) == 6
    assert len(engine.execution_log) == 0
