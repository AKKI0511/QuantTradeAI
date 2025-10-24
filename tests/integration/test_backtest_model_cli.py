import json
import os
import tempfile
from unittest.mock import patch
import yaml

import pandas as pd
from typer.testing import CliRunner


def _mock_data() -> pd.DataFrame:
    # 10-day OHLCV with two feature columns expected by the fake model
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(10, 20),
            "High": range(11, 21),
            "Low": range(9, 19),
            "Close": range(10, 20),
            "Volume": [1000] * 10,
            "f1": [0.1] * 10,
            "f2": [0.2] * 10,
        },
        index=idx,
    )
    return df


class FakeProcessor:
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_labels(self, df: pd.DataFrame, forward_returns: int = 1, threshold: float = 0.0) -> pd.DataFrame:
        df = df.copy()
        df["forward_returns"] = df["Close"].shift(-1) / df["Close"] - 1
        df["label"] = 0
        df.iloc[:-1, df.columns.get_loc("label")] = 1
        return df


class FakeClassifier:
    def __init__(self, *args, **kwargs):
        self.feature_columns = ["f1", "f2"]
        self.model = object()

    def load_model(self, path: str) -> None:
        return None

    def predict(self, X):
        # Deterministic alternating sequence
        import numpy as np

        n = X.shape[0]
        pattern = [1, 0, -1, 1]
        out = [pattern[i % len(pattern)] for i in range(n)]
        return np.array(out)


def test_backtest_model_cli_creates_artifacts_and_outputs_summary():
    # Import lazily to avoid Typer CLI initialization cost for unrelated tests
    from quanttradeai.cli import app

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare a minimal, valid model config aligned with mock data window
        model_cfg_path = os.path.join(tmpdir, "model_config.yaml")
        cfg = {
            "data": {
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
                "timeframe": "1d",
                "use_cache": False,
                # ensure non-empty test
                "test_start": "2020-01-05",
                "test_end": "2020-01-08",
            },
            "training": {"test_size": 0.2, "cv_folds": 3},
        }
        with open(model_cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        # Ensure model path exists
        model_dir = os.path.join(tmpdir, "fake_model")
        os.makedirs(model_dir, exist_ok=True)
        # Patch pipeline components to avoid network and heavy models
        with patch("quanttradeai.main.DataLoader.fetch_data", return_value={"AAA": _mock_data()}), \
            patch("quanttradeai.main.DataProcessor", FakeProcessor), \
            patch("quanttradeai.main.MomentumClassifier", FakeClassifier):

            # Run the CLI
            result = runner.invoke(
                app,
                [
                    "backtest-model",
                    "-m",
                    model_dir,
                    "-c",
                    model_cfg_path,
                ],
            )

    assert result.exit_code == 0, result.stdout
    # Parse the JSON summary and validate outputs exist
    summary = json.loads(result.stdout)
    assert "AAA" in summary
    out_dir = summary["AAA"]["output_dir"]
    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, "metrics.json"))
    assert os.path.isfile(os.path.join(out_dir, "equity_curve.csv"))
    # If ledger exists, ensure it has at least one row
    ledger_path = os.path.join(out_dir, "ledger.csv")
    if os.path.exists(ledger_path):
        import csv

        with open(ledger_path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) >= 2  # header + at least one trade


def test_backtest_model_cli_overrides_plumb_through():
    from quanttradeai.cli import app

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare model config and model path
        model_cfg_path = os.path.join(tmpdir, "model_config.yaml")
        cfg = {
            "data": {
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
                "timeframe": "1d",
                "use_cache": False,
                "test_start": "2020-01-05",
                "test_end": "2020-01-08",
            },
            "training": {"test_size": 0.2, "cv_folds": 3},
        }
        with open(model_cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        model_dir = os.path.join(tmpdir, "fake_model")
        os.makedirs(model_dir, exist_ok=True)
        with patch("quanttradeai.main.DataLoader.fetch_data", return_value={"AAA": _mock_data()}), \
            patch("quanttradeai.main.DataProcessor", FakeProcessor), \
            patch("quanttradeai.main.MomentumClassifier", FakeClassifier):

            result = runner.invoke(
                app,
                [
                    "backtest-model",
                    "-m",
                    model_dir,
                    "-c",
                    model_cfg_path,
                    "--cost-bps",
                    "5",
                    "--slippage-fixed",
                    "0.01",
                    "--liquidity-max-participation",
                    "0.5",
                ],
            )

    assert result.exit_code == 0, result.stdout
    # Ensure summary JSON contains the symbol and metrics
    summary = json.loads(result.stdout)
    assert "AAA" in summary
    aaa = summary["AAA"]
    # At least metrics should be present with Sharpe aliases
    assert "metrics" in aaa
    metrics = aaa["metrics"]
    assert any(k in metrics for k in ("sharpe_ratio", "gross_sharpe", "net_sharpe"))


def test_backtest_model_cli_risk_config_passed_to_drawdown_guard():
    from quanttradeai.cli import app

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_cfg_path = os.path.join(tmpdir, "model_config.yaml")
        cfg = {
            "data": {
                "symbols": ["AAA"],
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
                "timeframe": "1d",
                "use_cache": False,
                "test_start": "2020-01-05",
                "test_end": "2020-01-08",
            },
            "training": {"test_size": 0.2, "cv_folds": 3},
        }
        with open(model_cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        model_dir = os.path.join(tmpdir, "fake_model")
        os.makedirs(model_dir, exist_ok=True)
        risk_path = os.path.join(tmpdir, "risk.yaml")
        with open(risk_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"risk_management": {"drawdown_protection": {"max_drawdown_pct": 0.1}}}, f)

        fake_result = pd.DataFrame(
            {
                "strategy_return": [0.01, -0.005],
                "equity_curve": [1.01, 1.00495],
            },
            index=pd.date_range("2020-01-05", periods=2, freq="D"),
        )

        with patch("quanttradeai.main.DataLoader.fetch_data", return_value={"AAA": _mock_data()}), \
            patch("quanttradeai.main.DataProcessor", FakeProcessor), \
            patch("quanttradeai.main.MomentumClassifier", FakeClassifier), \
            patch("quanttradeai.main.DrawdownGuard") as mock_guard, \
            patch("quanttradeai.main.simulate_trades", return_value=fake_result) as mock_sim, \
            patch("quanttradeai.main.compute_metrics", return_value={"net": 1.0}):

            result = runner.invoke(
                app,
                [
                    "backtest-model",
                    "-m",
                    model_dir,
                    "-c",
                    model_cfg_path,
                    "--risk-config",
                    risk_path,
                ],
            )

    assert result.exit_code == 0, result.stdout
    mock_guard.assert_called_once()
    assert mock_guard.call_args.kwargs.get("config_path") == risk_path
    assert mock_sim.call_args.kwargs.get("drawdown_guard") is mock_guard.return_value

