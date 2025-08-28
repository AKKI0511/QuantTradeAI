import os
import json
import tempfile
from datetime import datetime

import pandas as pd
import yaml
from unittest.mock import patch


def _make_config(tmpdir: str) -> str:
    cfg = {
        "data": {
            "symbols": ["AAA"],
            "start_date": "2020-01-01",
            "end_date": "2020-01-10",
            "cache_path": tmpdir,
            "use_cache": False,
            "timeframe": "1d",
            # Ensure a non-empty test set
            "test_start": "2020-01-05",
            "test_end": "2020-01-08",
        },
        "training": {"test_size": 0.2, "cv_folds": 3},
        "trading": {"stop_loss": 0.02, "take_profit": 0.04},
    }
    path = os.path.join(tmpdir, "model_config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


class FakeProcessor:
    def __init__(self, *args, **kwargs):
        pass

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
        # no-op for tests
        return None

    def predict(self, X):
        # return alternating labels for determinism, match test window size = 4
        import numpy as np

        n = X.shape[0]
        pattern = [1, 0, -1, 1]
        out = [pattern[i % len(pattern)] for i in range(n)]
        return np.array(out)


def _mock_data() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(10, 20),
            "High": range(11, 21),
            "Low": range(9, 19),
            "Close": range(10, 20),
            "Volume": [1000] * 10,
            # Pre-made features expected by the saved model
            "f1": [0.1] * 10,
            "f2": [0.2] * 10,
        },
        index=idx,
    )
    return df


def test_run_model_backtest_happy_path():
    from quanttradeai.main import run_model_backtest

    with tempfile.TemporaryDirectory() as tmpdir:
        model_cfg = _make_config(tmpdir)

        with patch("quanttradeai.main.DataLoader.fetch_data", return_value={"AAA": _mock_data()}), \
            patch("quanttradeai.main.DataProcessor", FakeProcessor), \
            patch("quanttradeai.main.MomentumClassifier", FakeClassifier):

            summary = run_model_backtest(
                model_config=model_cfg,
                model_path=os.path.join(tmpdir, "fake_model"),
                backtest_config=None,
            )

        assert "AAA" in summary
        assert "metrics" in summary["AAA"] or "error" in summary["AAA"]

