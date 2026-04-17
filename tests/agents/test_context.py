import pandas as pd

from quanttradeai.agents.base import AgentSimulationState
from quanttradeai.agents.context import build_context_payload


def _history_with_news() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [101.0, 102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.5, 101.5, 102.5, 103.5],
            "Volume": [1000.0, 1100.0, 1200.0, 1300.0],
            "rsi": [45.0, 50.0, 55.0, 60.0],
            "text": [
                "",
                "Federal Reserve commentary eases rate fears",
                "Federal Reserve commentary eases rate fears",
                "Apple launches new buyback program",
            ],
        },
        index=index,
    )


def test_build_context_payload_adds_prompt_only_blocks_for_llm_agents():
    history = _history_with_news()
    current_row = history.iloc[-1]
    agent_config = {
        "name": "breakout_gpt",
        "kind": "llm",
        "_current_symbol": "AAPL",
        "context": {
            "features": ["rsi_14"],
            "orders": {"enabled": True, "max_entries": 2},
            "memory": {"enabled": True, "max_entries": 2},
            "news": {"enabled": True, "max_items": 2},
            "notes": True,
        },
    }
    decision_history = [
        {
            "symbol": "MSFT",
            "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
            "action": "buy",
            "reason": "ignore other symbols",
            "execution_status": "executed",
            "target_position_after": 1,
        },
        {
            "symbol": "AAPL",
            "timestamp": pd.Timestamp("2024-01-02T00:00:00Z"),
            "action": "buy",
            "reason": "breakout confirmed",
            "execution_status": "executed",
            "target_position_after": 1,
        },
        {
            "symbol": "AAPL",
            "timestamp": pd.Timestamp("2024-01-03T00:00:00Z"),
            "action": "hold",
            "reason": "trend intact",
            "execution_status": "no_change",
            "target_position_after": 1,
        },
    ]
    execution_history = [
        {
            "symbol": "MSFT",
            "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
            "action": "buy",
            "qty": 1,
            "price": 99.0,
            "status": "executed",
        },
        {
            "symbol": "AAPL",
            "timestamp": pd.Timestamp("2024-01-02T00:00:00Z"),
            "action": "buy",
            "qty": 1,
            "price": 101.5,
            "status": "executed",
        },
        {
            "symbol": "AAPL",
            "timestamp": pd.Timestamp("2024-01-03T00:00:00Z"),
            "action": "sell",
            "qty": 2,
            "price": 102.5,
            "status": "simulated",
        },
    ]

    payload = build_context_payload(
        feature_definitions=[
            {"name": "rsi_14", "type": "technical", "params": {"period": 14}}
        ],
        agent_config=agent_config,
        history=history,
        current_row=current_row,
        model_signals={},
        state=AgentSimulationState(),
        decision_history=decision_history,
        execution_history=execution_history,
        notes_payload={"path": "notes/breakout_gpt.md", "content": "Trade the trend."},
    )

    assert payload["orders"]["recent_orders"] == [
        {
            "timestamp": "2024-01-03T00:00:00+00:00",
            "action": "sell",
            "qty": 2,
            "price": 102.5,
            "status": "simulated",
        },
        {
            "timestamp": "2024-01-02T00:00:00+00:00",
            "action": "buy",
            "qty": 1,
            "price": 101.5,
            "status": "executed",
        },
    ]
    assert payload["memory"]["recent_decisions"] == [
        {
            "timestamp": "2024-01-03T00:00:00+00:00",
            "action": "hold",
            "reason": "trend intact",
            "execution_status": "no_change",
            "target_position_after": 1,
        },
        {
            "timestamp": "2024-01-02T00:00:00+00:00",
            "action": "buy",
            "reason": "breakout confirmed",
            "execution_status": "executed",
            "target_position_after": 1,
        },
    ]
    assert payload["news"]["headlines"] == [
        {
            "timestamp": "2024-01-04T00:00:00+00:00",
            "text": "Apple launches new buyback program",
        },
        {
            "timestamp": "2024-01-03T00:00:00+00:00",
            "text": "Federal Reserve commentary eases rate fears",
        },
    ]
    assert payload["notes"] == {
        "path": "notes/breakout_gpt.md",
        "content": "Trade the trend.",
    }


def test_build_context_payload_keeps_enabled_empty_blocks_stable():
    history = _history_with_news().drop(columns=["text"])
    current_row = history.iloc[-1]
    agent_config = {
        "name": "breakout_gpt",
        "kind": "hybrid",
        "_current_symbol": "AAPL",
        "context": {
            "orders": True,
            "memory": True,
            "news": True,
        },
    }

    payload = build_context_payload(
        feature_definitions=[],
        agent_config=agent_config,
        history=history,
        current_row=current_row,
        model_signals={},
        state=AgentSimulationState(),
        decision_history=[],
        execution_history=[],
    )

    assert payload["orders"] == {"recent_orders": []}
    assert payload["memory"] == {"recent_decisions": []}
    assert payload["news"] == {"headlines": []}
