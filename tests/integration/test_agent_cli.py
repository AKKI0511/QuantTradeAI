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


class FakeSignalClassifier:
    def __init__(self, *args, **kwargs):
        self.feature_columns = ["rsi"]

    def load_model(self, path: str) -> None:
        return None

    def predict(self, X):
        return np.array([1 for _ in range(len(X))])


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
