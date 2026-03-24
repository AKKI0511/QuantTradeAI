import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from typer.testing import CliRunner

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


def _fake_generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["rsi"] = np.linspace(45.0, 65.0, len(featured))
    featured["volume_momentum_20"] = np.linspace(0.1, 0.3, len(featured))
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

    decision_lines = (
        (run_dir / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert decision_lines
    first_decision = json.loads(decision_lines[0])
    assert first_decision["action"] == "buy"

    prompt_samples = json.loads((run_dir / "prompt_samples.json").read_text("utf-8"))
    assert prompt_samples
    assert "messages" in prompt_samples[0]["prompt_payload"]


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
    assert "ledger" not in payload["artifacts"]
    assert not (run_dir / "ledger.csv").exists()


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
            position["qty"] * position["price"]
            for position in self.positions.values()
        )


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


def test_hybrid_agent_run_includes_model_signals(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    init_result = runner.invoke(
        app,
        ["init", "--template", "hybrid", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    model_dir = Path("models/trained/aapl_daily_classifier")
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path("config/project.yaml")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["start_date"] = "2024-01-01"
    config_payload["data"]["end_date"] = "2024-02-09"
    config_payload["data"]["test_start"] = "2024-01-26"
    config_payload["data"]["test_end"] = "2024-01-31"
    config_payload["agents"][0]["model_signal_sources"] = [
        {
            "name": "aapl_daily_classifier",
            "path": "models/trained/aapl_daily_classifier",
        }
    ]
    config_payload["agents"][0]["context"]["model_signals"] = ["aapl_daily_classifier"]
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


def test_model_agent_backtest_writes_standardized_artifacts(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

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
    assert "prompt_samples" not in payload["artifacts"]

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

    init_result = runner.invoke(
        app,
        ["init", "--template", "model-agent", "--output", "config/project.yaml"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    observed: dict[str, str] = {}

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
