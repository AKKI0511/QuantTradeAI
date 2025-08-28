import json
import os
import tempfile
from unittest.mock import patch

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
                    os.path.join(tmpdir, "fake_model"),
                    "-c",
                    os.path.join(tmpdir, "model_config.yaml"),  # not used by fakes
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
        with patch("quanttradeai.main.DataLoader.fetch_data", return_value={"AAA": _mock_data()}), \
            patch("quanttradeai.main.DataProcessor", FakeProcessor), \
            patch("quanttradeai.main.MomentumClassifier", FakeClassifier):

            result = runner.invoke(
                app,
                [
                    "backtest-model",
                    "-m",
                    os.path.join(tmpdir, "fake_model"),
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

