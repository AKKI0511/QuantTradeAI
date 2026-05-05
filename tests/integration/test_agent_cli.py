import json
import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from typer.testing import CliRunner

from quanttradeai.brokers.base import (
    BrokerAccountSnapshot,
    BrokerOrderResult,
    BrokerPositionSnapshot,
)
from quanttradeai.cli import app


runner = CliRunner()


def _mock_history() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=40, freq="D")
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


def _mock_history_with_news() -> pd.DataFrame:
    history = _mock_history()
    history["text"] = pd.Series(
        [np.nan] * len(history), index=history.index, dtype=object
    )
    history.loc[pd.Timestamp("2024-01-24"), "text"] = "Apple expands its buyback"
    history.loc[pd.Timestamp("2024-01-25"), "text"] = "Apple expands its buyback"
    history.loc[pd.Timestamp("2024-01-26"), "text"] = "Analyst raises Apple target"
    return history


def _fake_generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["rsi"] = np.linspace(45.0, 65.0, len(featured))
    featured["volume_momentum_20"] = np.linspace(0.1, 0.3, len(featured))
    return featured


def _drop_text_sensitive_generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    if "text" in featured.columns:
        featured = featured.dropna()
    featured["rsi"] = np.linspace(45.0, 65.0, len(featured))
    featured["volume_momentum_20"] = np.linspace(0.1, 0.3, len(featured))
    return featured


def _rule_backtest_features(self, df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    rsi_values = np.full(len(featured), 55.0)
    index = pd.to_datetime(featured.index)
    for timestamp, value in {
        pd.Timestamp("2024-01-26"): 45.0,
        pd.Timestamp("2024-01-27"): 48.0,
        pd.Timestamp("2024-01-28"): 52.0,
        pd.Timestamp("2024-01-29"): 58.0,
        pd.Timestamp("2024-01-30"): 62.0,
        pd.Timestamp("2024-01-31"): 64.0,
    }.items():
        matches = index == timestamp
        if matches.any():
            rsi_values[matches] = value
    featured["rsi"] = rsi_values
    return featured


def _rule_paper_features(self, df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["rsi"] = 55.0
    for timestamp, value in {
        pd.Timestamp("2024-02-10T00:00:00Z"): 45.0,
        pd.Timestamp("2024-02-11T00:00:00Z"): 65.0,
        pd.Timestamp("2024-02-12T00:00:00Z"): 55.0,
    }.items():
        if timestamp in featured.index:
            featured.loc[timestamp, "rsi"] = value
    return featured


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


def _stub_prometheus_client(monkeypatch) -> None:
    return None


def _stub_live_trading_module(monkeypatch) -> None:
    live_trading_module = types.ModuleType("quanttradeai.streaming.live_trading")
    live_trading_module.LiveTradingEngine = object
    monkeypatch.setitem(
        sys.modules,
        "quanttradeai.streaming.live_trading",
        live_trading_module,
    )
    sys.modules.pop("quanttradeai.agents.model_agent", None)


def _stub_streaming_gateway_module(monkeypatch) -> None:
    gateway_module = types.ModuleType("quanttradeai.streaming.gateway")
    gateway_module.StreamingGateway = object
    monkeypatch.setitem(
        sys.modules,
        "quanttradeai.streaming.gateway",
        gateway_module,
    )
    sys.modules.pop("quanttradeai.agents.paper", None)


def test_agent_run_backtest_writes_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    init_result = runner.invoke(
        app,
        ["init", "--template", "llm-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-01-26"
    config_payload["data"]["test_end"] = "2024-01-31"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    with (
        patch(
            "quanttradeai.agents.backtest.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.backtest.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            side_effect=_completion_from_actions(
                ["buy", "hold", "sell", "hold", "buy"]
            ),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "breakout_gpt",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
                "--skip-validation",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "success"
    assert payload["run_type"] == "agent"
    assert payload["name"] == "breakout_gpt"
    assert payload["run_id"].startswith("agent/backtest/")
    assert run_dir.parts[:3] == ("runs", "agent", "backtest")
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "equity_curve.csv").is_file()
    assert (run_dir / "decisions.jsonl").is_file()
    assert (run_dir / "prompt_samples.json").is_file()
    assert not (run_dir / "run_brief.json").exists()
    assert not (run_dir / "run_brief.md").exists()
    summary_payload = json.loads((run_dir / "summary.json").read_text("utf-8"))
    assert summary_payload["run_result"]["next_action"]["action"] == (
        "promote_agent_to_paper"
    )
    assert "promote_to_paper" in summary_payload["run_result"]["commands"]

    decision_lines = (
        (run_dir / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert decision_lines
    first_decision = json.loads(decision_lines[0])
    assert first_decision["action"] == "buy"

    prompt_samples = json.loads((run_dir / "prompt_samples.json").read_text("utf-8"))
    assert prompt_samples
    assert "messages" in prompt_samples[0]["prompt_payload"]


def test_llm_agent_backtest_context_blocks_include_orders_memory_news_and_notes(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    init_result = runner.invoke(
        app,
        ["init", "--template", "llm-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-01-26"
    config_payload["data"]["test_end"] = "2024-01-31"
    config_payload["news"] = {"enabled": True}
    config_payload["agents"][0]["context"].update(
        {
            "orders": {"enabled": True, "max_entries": 2},
            "memory": {"enabled": True, "max_entries": 2},
            "news": {"enabled": True, "max_items": 2},
            "notes": True,
        }
    )
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    notes_path = Path("notes/breakout_gpt.md")
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("Favor continuation setups over noise.", encoding="utf-8")

    with (
        patch(
            "quanttradeai.agents.backtest.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history_with_news()},
        ),
        patch(
            "quanttradeai.agents.backtest.DataProcessor.generate_features",
            _drop_text_sensitive_generate_features,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            side_effect=_completion_from_actions(
                ["buy", "hold", "sell", "hold", "buy", "hold"]
            ),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "breakout_gpt",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
                "--skip-validation",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    decisions = [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    ]

    first_context = decisions[0]["context"]
    assert first_context["orders"] == {"recent_orders": []}
    assert first_context["memory"] == {"recent_decisions": []}
    assert first_context["notes"] == {
        "path": "notes/breakout_gpt.md",
        "content": "Favor continuation setups over noise.",
    }
    assert (
        first_context["news"]["headlines"][0]["text"] == "Analyst raises Apple target"
    )

    second_context = decisions[1]["context"]
    assert second_context["orders"]["recent_orders"][0]["action"] == "buy"
    assert second_context["orders"]["recent_orders"][0]["status"] == "simulated"
    assert second_context["memory"]["recent_decisions"][0]["action"] == "buy"
    assert (
        second_context["memory"]["recent_decisions"][0]["execution_status"]
        == "simulated"
    )


def test_agent_run_backtest_omits_ledger_artifact_when_no_trades(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    init_result = runner.invoke(
        app,
        ["init", "--template", "llm-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-01-26"
    config_payload["data"]["test_end"] = "2024-01-31"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    with (
        patch(
            "quanttradeai.agents.backtest.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.backtest.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            side_effect=_completion_from_actions(
                ["hold", "hold", "hold", "hold", "hold"]
            ),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "breakout_gpt",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
                "--skip-validation",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "success"
    assert "ledger" not in payload["important_artifacts"]
    assert not (run_dir / "ledger.csv").exists()


def test_agent_run_reports_project_config_load_errors(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    missing_config = Path("config/missing.yaml")
    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--agent",
            "breakout_gpt",
            "--config",
            str(missing_config),
            "--mode",
            "backtest",
        ],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "not found in project config" not in combined_output
    assert "Agent run failed:" in combined_output
    assert str(missing_config) in combined_output


class FakeSignalClassifier:
    def __init__(self, *args, **kwargs):
        self.feature_columns = ["rsi"]
        self.model = True

    def load_model(self, path: str) -> None:
        return None

    def predict(self, X):
        return np.array([1 for _ in range(len(X))])


class FakePaperPortfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = 95_000.0
        self.realized_pnl = 120.0
        self.positions = {
            "AAPL": {
                "qty": 10,
                "price": 105.0,
                "entry_price": 100.0,
                "stop_loss_pct": 0.02,
            }
        }

    @property
    def portfolio_value(self) -> float:
        return self.cash + sum(
            position["qty"] * position["price"] for position in self.positions.values()
        )

    def replace_state(
        self,
        *,
        cash: float,
        positions: dict,
        initial_capital: float | None = None,
        realized_pnl: float | None = None,
    ) -> None:
        self.cash = float(cash)
        self.positions = dict(positions)
        if initial_capital is not None:
            self.initial_capital = float(initial_capital)
        if realized_pnl is not None:
            self.realized_pnl = float(realized_pnl)

    def estimate_open_position_qty(
        self,
        price: float,
        stop_loss_pct: float | None = None,
        *,
        check_risk: bool = True,
    ) -> int:
        return max(int(self.cash // price), 0)


class FakePaperEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.execution_log = []
        self.portfolio = FakePaperPortfolio(kwargs["initial_capital"])
        self._history = {"AAPL": _mock_history().tail(2)}

    async def start(self) -> None:
        self.execution_log = [
            {
                "action": "buy",
                "symbol": "AAPL",
                "qty": 10,
                "price": 100.0,
                "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                "signal": 1,
            },
            {
                "action": "sell",
                "symbol": "AAPL",
                "qty": 5,
                "price": 104.0,
                "timestamp": pd.Timestamp("2024-01-01T00:01:00Z"),
                "signal": -1,
            },
        ]


class FakeDrawdownGuard:
    def check_drawdown_limits(self) -> dict[str, object]:
        return {"status": "ok", "halted": False}


class FakeRiskManager:
    def __init__(self) -> None:
        self.drawdown_guard = FakeDrawdownGuard()

    def get_risk_metrics(self) -> dict[str, float]:
        return {
            "current_drawdown_pct": 0.01,
            "current_drawdown_abs": 100.0,
        }


class FakeLiveEngine(FakePaperEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.risk_manager = FakeRiskManager()
        self.decision_log = []

    async def start(self) -> None:
        self.decision_log = [
            {
                "symbol": "AAPL",
                "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                "signal": 1,
                "action": "buy",
                "source": "model",
            }
        ]
        self.execution_log = [
            {
                "action": "buy",
                "symbol": "AAPL",
                "qty": 10,
                "price": 100.0,
                "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                "signal": 1,
            }
        ]


class FakeStreamBuffer:
    def __init__(self, maxsize: int) -> None:
        self.queue = asyncio.Queue(maxsize)

    async def put(self, item: dict) -> None:
        await self.queue.put(item)

    async def get(self) -> dict:
        return await self.queue.get()


class FakeStreamingGateway:
    def __init__(self, messages: list[dict]) -> None:
        self.buffer = FakeStreamBuffer(32)
        self.messages = messages

    async def _start(self) -> None:
        for message in self.messages:
            await self.buffer.put(message)


class FakeBrokerClient:
    provider = "alpaca"

    def __init__(self, starting_cash: float = 100000.0) -> None:
        self.cash = starting_cash
        self.qty = 0
        self.market_price = 0.0
        self.avg_entry_price = 0.0
        self.next_order_id = 1
        self.last_order = None

    def get_account(self) -> BrokerAccountSnapshot:
        equity = self.cash + (self.qty * self.market_price)
        return BrokerAccountSnapshot(
            account_id="acct-integration",
            cash=self.cash,
            equity=equity,
            buying_power=self.cash,
        )

    def list_positions(self) -> list[BrokerPositionSnapshot]:
        if self.qty <= 0:
            return []
        return [
            BrokerPositionSnapshot(
                symbol="AAPL",
                qty=self.qty,
                market_price=self.market_price,
                avg_entry_price=self.avg_entry_price,
            )
        ]

    def submit_market_order(
        self,
        *,
        symbol: str,
        action: str,
        qty: int,
    ) -> BrokerOrderResult:
        order_id = f"ord-{self.next_order_id}"
        self.next_order_id += 1
        self.last_order = {
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "qty": qty,
        }
        return BrokerOrderResult(
            order_id=order_id,
            symbol=symbol,
            action=action,
            qty=qty,
            status="new",
        )

    def get_order(self, order_id: str) -> BrokerOrderResult:
        raise AssertionError("FakeBrokerClient.get_order should not be called directly")

    def wait_for_order(
        self,
        order_id: str,
        *,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> BrokerOrderResult:
        assert self.last_order is not None
        qty = int(self.last_order["qty"])
        action = str(self.last_order["action"])
        fill_price = 121.0 if action == "buy" else 123.0
        if action == "buy":
            self.cash -= qty * fill_price
            self.qty += qty
            self.avg_entry_price = fill_price
            self.market_price = fill_price
        else:
            self.cash += qty * fill_price
            self.qty = 0
            self.avg_entry_price = 0.0
            self.market_price = fill_price
        return BrokerOrderResult(
            order_id=order_id,
            symbol=str(self.last_order["symbol"]),
            action=action,
            qty=qty,
            status="filled",
            filled_qty=qty,
            filled_avg_price=fill_price,
        )


def test_hybrid_agent_run_includes_model_signals(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    init_result = runner.invoke(
        app,
        ["init", "--template", "hybrid", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    model_dir = Path("models/promoted/aapl_daily_classifier")
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-01-26"
    config_payload["data"]["test_end"] = "2024-01-31"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    def _hybrid_completion(**kwargs):
        content = kwargs["messages"][-1]["content"]
        action = "buy" if '"signal": 1' in content else "hold"
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"action": action, "reason": "Model signal driven"}
                        )
                    }
                }
            ]
        }

    with (
        patch(
            "quanttradeai.agents.backtest.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.backtest.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.backtest.MomentumClassifier",
            FakeSignalClassifier,
        ),
        patch(
            "quanttradeai.agents.backtest._load_feature_preprocessor",
            return_value=None,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            side_effect=_hybrid_completion,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "hybrid_swing_agent",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
                "--skip-validation",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    decision_lines = (
        (run_dir / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert decision_lines
    first_decision = json.loads(decision_lines[0])
    assert first_decision["model_signals"]["aapl_daily_classifier"]["signal"] == 1
    assert first_decision["action"] == "buy"


def test_rule_agent_backtest_writes_standardized_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    init_result = runner.invoke(
        app,
        ["init", "--template", "rule-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-01-26"
    config_payload["data"]["test_end"] = "2024-01-31"
    config_payload["agents"][0]["rule"]["buy_below"] = 50.0
    config_payload["agents"][0]["rule"]["sell_above"] = 60.0
    config_payload["agents"][0]["context"]["notes"] = True
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    with (
        patch(
            "quanttradeai.agents.backtest.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.backtest.DataProcessor.generate_features",
            _rule_backtest_features,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "rsi_reversion",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
                "--skip-validation",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "success"
    assert payload["agent_kind"] == "rule"
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "equity_curve.csv").is_file()
    assert (run_dir / "decisions.jsonl").is_file()
    assert "prompt_samples" not in payload["important_artifacts"]
    assert not (run_dir / "prompt_samples.json").exists()

    decision_lines = (
        (run_dir / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    actions = [json.loads(line)["action"] for line in decision_lines]
    assert actions[:2] == ["buy", "buy"]
    assert "sell" in actions

    runs_result = runner.invoke(app, ["runs", "list", "--json"])
    assert runs_result.exit_code == 0, runs_result.stdout
    run_records = json.loads(runs_result.stdout)
    assert run_records[0]["name"] == "rsi_reversion"
    assert run_records[0]["run_type"] == "agent"


def test_model_agent_backtest_writes_standardized_artifacts(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _stub_prometheus_client(monkeypatch)
    _stub_live_trading_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "model-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-01-26"
    config_payload["data"]["test_end"] = "2024-01-31"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    with (
        patch(
            "quanttradeai.agents.model_agent.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.model_agent.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.model_agent.MomentumClassifier",
            FakeSignalClassifier,
        ),
        patch(
            "quanttradeai.agents.model_agent._load_feature_preprocessor",
            return_value=None,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "paper_momentum",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
                "--skip-validation",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "success"
    assert payload["agent_kind"] == "model"
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "equity_curve.csv").is_file()
    assert (run_dir / "decisions.jsonl").is_file()
    assert (run_dir / "runtime_backtest_config.yaml").is_file()
    assert "prompt_samples" not in payload["important_artifacts"]

    decision_lines = (
        (run_dir / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert decision_lines
    first_decision = json.loads(decision_lines[0])
    assert first_decision["action"] == "buy"
    assert first_decision["target_position"] == 1


def test_model_agent_paper_run_writes_metrics_and_execution_log(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _stub_prometheus_client(monkeypatch)
    _stub_live_trading_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "model-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    observed: dict[str, str] = {}
    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["streaming"]["replay"]["enabled"] = False
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    def _engine_factory(**kwargs):
        observed.update({key: str(value) for key, value in kwargs.items()})
        return FakePaperEngine(**kwargs)

    with patch(
        "quanttradeai.agents.model_agent.LiveTradingEngine",
        side_effect=_engine_factory,
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "paper_momentum",
                "--config",
                "config/project.yaml",
                "--mode",
                "paper",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "success"
    assert payload["mode"] == "paper"
    assert payload["agent_kind"] == "model"
    assert (run_dir / "executions.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "runtime_streaming_config.yaml").is_file()
    assert Path(observed["features_config"]).is_file()
    assert Path(observed["streaming_config"]).is_file()

    metrics_payload = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics_payload["execution_count"] == 2
    assert metrics_payload["realized_pnl"] == 120.0
    assert metrics_payload["open_positions"]["AAPL"]["unrealized_pnl"] == 50.0


def test_model_agent_paper_run_uses_replay_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub_prometheus_client(monkeypatch)
    _stub_live_trading_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "model-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-03"
    config_payload["data"]["streaming"].pop("websocket_url")
    config_payload["data"]["streaming"].pop("provider")
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    observed: dict[str, object] = {}

    def _engine_factory(**kwargs):
        observed.update(kwargs)
        return FakePaperEngine(**kwargs)

    with (
        patch(
            "quanttradeai.agents.model_agent.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.model_agent.LiveTradingEngine",
            side_effect=_engine_factory,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "paper_momentum",
                "--config",
                "config/project.yaml",
                "--mode",
                "paper",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["paper_source"] == "replay"
    assert (run_dir / "replay_manifest.json").is_file()
    assert not (run_dir / "run_brief.json").exists()
    summary_payload = json.loads((run_dir / "summary.json").read_text("utf-8"))
    assert summary_payload["run_result"]["next_action"]["action"] == (
        "promote_agent_to_live"
    )
    assert "promote_to_live" in summary_payload["run_result"]["commands"]
    assert observed["gateway"].__class__.__name__ == "ReplayGateway"
    assert observed["bootstrap_history_frames"]


def test_model_agent_live_run_writes_live_artifacts_and_risk_metrics(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _stub_prometheus_client(monkeypatch)
    _stub_live_trading_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "model-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["agents"][0]["mode"] = "live"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    observed: dict[str, str] = {}

    def _engine_factory(**kwargs):
        observed.update({key: str(value) for key, value in kwargs.items()})
        return FakeLiveEngine(**kwargs)

    with patch(
        "quanttradeai.agents.model_agent.LiveTradingEngine",
        side_effect=_engine_factory,
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "paper_momentum",
                "--config",
                str(config_path),
                "--mode",
                "live",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "success"
    assert payload["mode"] == "live"
    assert payload["agent_kind"] == "model"
    assert (run_dir / "runtime_streaming_config.yaml").is_file()
    assert (run_dir / "runtime_risk_config.yaml").is_file()
    assert (run_dir / "runtime_position_manager_config.yaml").is_file()
    assert (run_dir / "decisions.jsonl").is_file()
    assert (run_dir / "executions.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert Path(observed["risk_config"]).is_file()
    assert Path(observed["position_manager_config"]).is_file()

    metrics_payload = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics_payload["decision_count"] == 1
    assert metrics_payload["execution_count"] == 1
    assert metrics_payload["risk_metrics"]["current_drawdown_pct"] == 0.01
    assert metrics_payload["risk_status"]["status"] == "ok"


def test_model_agent_live_run_with_alpaca_backend_writes_broker_artifacts(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _stub_prometheus_client(monkeypatch)
    _stub_live_trading_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "model-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["agents"][0]["mode"] = "live"
    config_payload["agents"][0]["execution"] = {"backend": "alpaca"}
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    class FakeBrokerAwareLiveEngine(FakeLiveEngine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.position_manager = None

        @property
        def broker_provider(self):
            runtime = getattr(self, "broker_runtime", None)
            return runtime.provider if runtime is not None else None

        async def start(self) -> None:
            self.decision_log = []
            self.execution_log = []
            if getattr(self, "broker_runtime", None) is not None:
                self.broker_runtime.start_session()
                execution_status, execution_payload = (
                    self.broker_runtime.execute_action(
                        symbol="AAPL",
                        action="buy",
                        price=100.0,
                        timestamp=pd.Timestamp("2024-01-01T00:00:00Z").to_pydatetime(),
                        extra={"signal": 1},
                    )
                )
                self.execution_log = [execution_payload]
                self.decision_log = [
                    {
                        "symbol": "AAPL",
                        "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                        "signal": 1,
                        "action": "buy",
                        "source": "model",
                        "execution_status": execution_status,
                        "execution": execution_payload,
                        "target_position": 1,
                    }
                ]
                self.broker_runtime.finish_session()
            else:
                await super().start()

    with (
        patch(
            "quanttradeai.agents.model_agent.LiveTradingEngine",
            side_effect=lambda **kwargs: FakeBrokerAwareLiveEngine(**kwargs),
        ),
        patch(
            "quanttradeai.agents.model_agent.create_broker_client_for_agent",
            return_value=FakeBrokerClient(),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "paper_momentum",
                "--config",
                str(config_path),
                "--mode",
                "live",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["execution_backend"] == "alpaca"
    assert payload["broker_provider"] == "alpaca"
    assert (run_dir / "broker_account_start.json").is_file()
    assert (run_dir / "broker_account_end.json").is_file()

    execution_lines = [
        json.loads(line)
        for line in (run_dir / "executions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    ]
    assert execution_lines[0]["order_id"] == "ord-1"
    assert execution_lines[0]["status"] == "filled"


def test_llm_agent_paper_run_writes_standardized_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub_streaming_gateway_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "llm-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-09"
    config_payload["agents"][0]["mode"] = "paper"
    config_payload["data"]["streaming"]["replay"]["enabled"] = False
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    gateway = FakeStreamingGateway(
        [
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-02-10T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 123.0,
                "volume": 12,
                "timestamp": "2024-02-11T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 124.0,
                "volume": 14,
                "timestamp": "2024-02-12T00:00:00Z",
            },
        ]
    )

    with (
        patch(
            "quanttradeai.agents.paper.StreamingGateway",
            return_value=gateway,
        ),
        patch(
            "quanttradeai.agents.paper.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.paper.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            side_effect=_completion_from_actions(["buy", "hold"]),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "breakout_gpt",
                "--config",
                str(config_path),
                "--mode",
                "paper",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "success"
    assert payload["mode"] == "paper"
    assert payload["agent_kind"] == "llm"
    assert (run_dir / "resolved_project_config.yaml").is_file()
    assert (run_dir / "runtime_model_config.yaml").is_file()
    assert (run_dir / "runtime_features_config.yaml").is_file()
    assert (run_dir / "runtime_streaming_config.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "decisions.jsonl").is_file()
    assert (run_dir / "executions.jsonl").is_file()
    assert (run_dir / "prompt_samples.json").is_file()

    decision_lines = (
        (run_dir / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    execution_lines = (
        (run_dir / "executions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(decision_lines) == 2
    assert len(execution_lines) == 1

    metrics_payload = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics_payload["decision_count"] == 2
    assert metrics_payload["execution_count"] == 1


def test_llm_agent_paper_context_blocks_include_orders_memory_news_and_notes(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub_streaming_gateway_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "llm-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-09"
    config_payload["agents"][0]["mode"] = "paper"
    config_payload["data"]["streaming"]["replay"]["enabled"] = False
    config_payload["news"] = {"enabled": True}
    config_payload["agents"][0]["context"].update(
        {
            "orders": {"enabled": True, "max_entries": 2},
            "memory": {"enabled": True, "max_entries": 2},
            "news": {"enabled": True, "max_items": 2},
            "notes": True,
        }
    )
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    notes_path = Path("notes/breakout_gpt.md")
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("Bias toward clear trend continuation.", encoding="utf-8")

    gateway = FakeStreamingGateway(
        [
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-02-10T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 123.0,
                "volume": 12,
                "timestamp": "2024-02-11T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 124.0,
                "volume": 14,
                "timestamp": "2024-02-12T00:00:00Z",
            },
        ]
    )

    with (
        patch(
            "quanttradeai.agents.paper.StreamingGateway",
            return_value=gateway,
        ),
        patch(
            "quanttradeai.agents.paper.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history_with_news()},
        ),
        patch(
            "quanttradeai.agents.paper.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            side_effect=_completion_from_actions(["buy", "hold"]),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "breakout_gpt",
                "--config",
                str(config_path),
                "--mode",
                "paper",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    decisions = [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    ]

    first_context = decisions[0]["context"]
    assert first_context["orders"] == {"recent_orders": []}
    assert first_context["memory"] == {"recent_decisions": []}
    assert first_context["notes"] == {
        "path": "notes/breakout_gpt.md",
        "content": "Bias toward clear trend continuation.",
    }
    assert (
        first_context["news"]["headlines"][0]["text"] == "Analyst raises Apple target"
    )

    second_context = decisions[1]["context"]
    assert second_context["orders"]["recent_orders"][0]["action"] == "buy"
    assert second_context["orders"]["recent_orders"][0]["status"] == "executed"
    assert second_context["memory"]["recent_decisions"][0]["action"] == "buy"
    assert (
        second_context["memory"]["recent_decisions"][0]["execution_status"]
        == "executed"
    )


def test_llm_agent_paper_run_uses_replay_without_realtime_streaming(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    init_result = runner.invoke(
        app,
        ["init", "--template", "llm-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-03"
    config_payload["data"]["streaming"].pop("websocket_url")
    config_payload["data"]["streaming"].pop("provider")
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    with (
        patch(
            "quanttradeai.agents.paper.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.paper.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            side_effect=_completion_from_actions(["buy", "hold", "sell", "hold"]),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "breakout_gpt",
                "--config",
                str(config_path),
                "--mode",
                "paper",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["paper_source"] == "replay"
    assert (run_dir / "replay_manifest.json").is_file()
    assert (run_dir / "prompt_samples.json").is_file()


def test_rule_agent_paper_run_writes_metrics_and_execution_log(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _stub_streaming_gateway_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "rule-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-09"
    config_payload["agents"][0]["rule"]["buy_below"] = 50.0
    config_payload["agents"][0]["rule"]["sell_above"] = 60.0
    config_payload["agents"][0]["context"]["notes"] = True
    config_payload["data"]["streaming"]["replay"]["enabled"] = False
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    gateway = FakeStreamingGateway(
        [
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-02-10T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 123.0,
                "volume": 12,
                "timestamp": "2024-02-11T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 124.0,
                "volume": 14,
                "timestamp": "2024-02-12T00:00:00Z",
            },
        ]
    )

    with (
        patch(
            "quanttradeai.agents.paper.StreamingGateway",
            return_value=gateway,
        ),
        patch(
            "quanttradeai.agents.paper.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.paper.DataProcessor.generate_features",
            _rule_paper_features,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "rsi_reversion",
                "--config",
                str(config_path),
                "--mode",
                "paper",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["status"] == "success"
    assert payload["agent_kind"] == "rule"
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "decisions.jsonl").is_file()
    assert (run_dir / "executions.jsonl").is_file()
    assert "prompt_samples" not in payload["important_artifacts"]
    assert not (run_dir / "prompt_samples.json").exists()

    decision_lines = [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    ]
    execution_lines = [
        json.loads(line)
        for line in (run_dir / "executions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    ]
    assert [entry["action"] for entry in decision_lines] == ["buy", "sell"]
    assert [entry["action"] for entry in execution_lines] == ["buy", "sell"]

    metrics_payload = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics_payload["decision_count"] == 2
    assert metrics_payload["execution_count"] == 2


def test_rule_agent_paper_run_with_alpaca_backend_writes_broker_artifacts(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _stub_streaming_gateway_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "rule-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-09"
    config_payload["data"]["streaming"]["replay"]["enabled"] = False
    config_payload["agents"][0]["execution"] = {"backend": "alpaca"}
    config_payload["agents"][0]["rule"]["buy_below"] = 50.0
    config_payload["agents"][0]["rule"]["sell_above"] = 60.0
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    gateway = FakeStreamingGateway(
        [
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-02-10T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 123.0,
                "volume": 12,
                "timestamp": "2024-02-11T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 124.0,
                "volume": 14,
                "timestamp": "2024-02-12T00:00:00Z",
            },
        ]
    )

    with (
        patch(
            "quanttradeai.agents.paper.StreamingGateway",
            return_value=gateway,
        ),
        patch(
            "quanttradeai.agents.paper.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.paper.DataProcessor.generate_features",
            _rule_paper_features,
        ),
        patch(
            "quanttradeai.agents.paper.create_broker_client_for_agent",
            return_value=FakeBrokerClient(),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "rsi_reversion",
                "--config",
                str(config_path),
                "--mode",
                "paper",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["execution_backend"] == "alpaca"
    assert payload["broker_provider"] == "alpaca"
    assert (run_dir / "broker_account_start.json").is_file()
    assert (run_dir / "broker_account_end.json").is_file()
    assert (run_dir / "broker_positions_start.json").is_file()
    assert (run_dir / "broker_positions_end.json").is_file()

    execution_lines = [
        json.loads(line)
        for line in (run_dir / "executions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    ]
    assert execution_lines[0]["order_id"] == "ord-1"
    assert execution_lines[0]["status"] == "filled"
    assert execution_lines[0]["filled_qty"] > 0

    metrics_payload = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics_payload["execution_backend"] == "alpaca"
    assert metrics_payload["broker_provider"] == "alpaca"


def test_rule_agent_paper_run_uses_replay_without_realtime_streaming(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    init_result = runner.invoke(
        app,
        ["init", "--template", "rule-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-03"
    config_payload["data"]["streaming"].pop("websocket_url")
    config_payload["data"]["streaming"].pop("provider")
    config_payload["agents"][0]["rule"]["buy_below"] = 50.0
    config_payload["agents"][0]["rule"]["sell_above"] = 60.0
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    with (
        patch(
            "quanttradeai.agents.paper.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.paper.DataProcessor.generate_features",
            _rule_paper_features,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "rsi_reversion",
                "--config",
                str(config_path),
                "--mode",
                "paper",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["paper_source"] == "replay"
    assert (run_dir / "replay_manifest.json").is_file()


def test_rule_agent_live_run_writes_live_artifacts_and_risk_metrics(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _stub_streaming_gateway_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "rule-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-09"
    config_payload["agents"][0]["mode"] = "live"
    config_payload["agents"][0]["rule"]["buy_below"] = 50.0
    config_payload["agents"][0]["rule"]["sell_above"] = 60.0
    config_payload["risk"]["drawdown_protection"]["max_drawdown_pct"] = 1.0
    config_payload["risk"]["turnover_limits"]["daily_max"] = 1_000_000.0
    config_payload["risk"]["turnover_limits"]["weekly_max"] = 1_000_000.0
    config_payload["risk"]["turnover_limits"]["monthly_max"] = 1_000_000.0
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    gateway = FakeStreamingGateway(
        [
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-02-10T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 123.0,
                "volume": 12,
                "timestamp": "2024-02-11T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 124.0,
                "volume": 14,
                "timestamp": "2024-02-12T00:00:00Z",
            },
        ]
    )

    with (
        patch(
            "quanttradeai.agents.paper.StreamingGateway",
            return_value=gateway,
        ),
        patch(
            "quanttradeai.agents.paper.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.paper.DataProcessor.generate_features",
            _rule_paper_features,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "rsi_reversion",
                "--config",
                str(config_path),
                "--mode",
                "live",
            ],
        )

    assert result.exit_code == 0, result.stdout
    run_dir = sorted((Path("runs") / "agent" / "live").iterdir())[-1]
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["mode"] == "live"
    assert payload["agent_kind"] == "rule"
    assert (run_dir / "runtime_streaming_config.yaml").is_file()
    assert (run_dir / "runtime_risk_config.yaml").is_file()
    assert (run_dir / "runtime_position_manager_config.yaml").is_file()
    assert (run_dir / "decisions.jsonl").is_file()
    assert (run_dir / "executions.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()

    metrics_payload = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics_payload["decision_count"] == 2
    assert metrics_payload["execution_count"] == 2
    assert "risk_metrics" in metrics_payload
    assert metrics_payload["risk_status"]["status"] in {
        "ok",
        "normal",
        "warning",
        "soft_stop",
        "hard_stop",
        "emergency_stop",
    }


def test_hybrid_agent_paper_run_includes_model_signals_in_decisions(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub_streaming_gateway_module(monkeypatch)

    init_result = runner.invoke(
        app,
        ["init", "--template", "hybrid", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    model_dir = Path("models/promoted/aapl_daily_classifier")
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-09"
    config_payload["agents"][0]["mode"] = "paper"
    config_payload["data"]["streaming"]["replay"]["enabled"] = False
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    gateway = FakeStreamingGateway(
        [
            {
                "symbol": "AAPL",
                "price": 121.0,
                "volume": 10,
                "timestamp": "2024-02-10T00:00:00Z",
            },
            {
                "symbol": "AAPL",
                "price": 123.0,
                "volume": 12,
                "timestamp": "2024-02-11T00:00:00Z",
            },
        ]
    )

    def _hybrid_completion(**kwargs):
        content = kwargs["messages"][-1]["content"]
        action = "buy" if '"signal": 1' in content else "hold"
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"action": action, "reason": "Model signal driven"}
                        )
                    }
                }
            ]
        }

    with (
        patch(
            "quanttradeai.agents.paper.StreamingGateway",
            return_value=gateway,
        ),
        patch(
            "quanttradeai.agents.paper.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.paper.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.backtest.MomentumClassifier",
            FakeSignalClassifier,
        ),
        patch(
            "quanttradeai.agents.backtest._load_feature_preprocessor",
            return_value=None,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            side_effect=_hybrid_completion,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "hybrid_swing_agent",
                "--config",
                str(config_path),
                "--mode",
                "paper",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    decision_lines = (
        (run_dir / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(decision_lines) == 1
    first_decision = json.loads(decision_lines[0])
    assert first_decision["model_signals"]["aapl_daily_classifier"]["signal"] == 1
    assert first_decision["action"] == "buy"


def test_hybrid_agent_paper_run_uses_replay_without_realtime_streaming(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    init_result = runner.invoke(
        app,
        ["init", "--template", "hybrid", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    model_dir = Path("models/promoted/aapl_daily_classifier")
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-03"
    config_payload["data"]["streaming"].pop("websocket_url")
    config_payload["data"]["streaming"].pop("provider")
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    def _hybrid_completion(**kwargs):
        content = kwargs["messages"][-1]["content"]
        action = "buy" if '"signal": 1' in content else "hold"
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"action": action, "reason": "Model signal driven"}
                        )
                    }
                }
            ]
        }

    with (
        patch(
            "quanttradeai.agents.paper.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.paper.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.backtest.MomentumClassifier",
            FakeSignalClassifier,
        ),
        patch(
            "quanttradeai.agents.backtest._load_feature_preprocessor",
            return_value=None,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            side_effect=_hybrid_completion,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "hybrid_swing_agent",
                "--config",
                str(config_path),
                "--mode",
                "paper",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    run_dir = Path(payload["run_dir"])
    assert payload["paper_source"] == "replay"
    assert (run_dir / "replay_manifest.json").is_file()


def test_agent_run_warns_when_cli_mode_differs_from_configured_mode(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    init_result = runner.invoke(
        app,
        ["init", "--template", "llm-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-02-01"
    config_payload["data"]["test_end"] = "2024-02-09"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    with (
        patch(
            "quanttradeai.agents.backtest.DataLoader.fetch_data",
            return_value={"AAPL": _mock_history()},
        ),
        patch(
            "quanttradeai.agents.backtest.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            side_effect=_completion_from_actions(
                ["hold", "hold", "hold", "hold", "hold"]
            ),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--agent",
                "breakout_gpt",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
                "--skip-validation",
            ],
        )

    assert result.exit_code == 0, result.stdout
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert (
        "configured with mode=paper but cli requested mode=backtest"
        in combined_output.lower()
    )


def test_agent_run_live_rejects_skip_validation(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    init_result = runner.invoke(
        app,
        ["init", "--template", "rule-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["agents"][0]["mode"] = "live"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--agent",
            "rsi_reversion",
            "--config",
            str(config_path),
            "--mode",
            "live",
            "--skip-validation",
        ],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "--skip-validation is not supported for live agent runs" in combined_output


def test_agent_run_live_requires_agent_to_be_configured_live(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    init_result = runner.invoke(
        app,
        ["init", "--template", "rule-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--agent",
            "rsi_reversion",
            "--config",
            "config/project.yaml",
            "--mode",
            "live",
        ],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "must be configured with mode=live before running" in combined_output
