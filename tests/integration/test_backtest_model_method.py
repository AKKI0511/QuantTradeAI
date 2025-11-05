import os
import tempfile

import pandas as pd
import yaml
from unittest.mock import patch

from quanttradeai.trading.portfolio import PortfolioManager


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
        # Ensure model directory exists
        model_dir = os.path.join(tmpdir, "fake_model")
        os.makedirs(model_dir, exist_ok=True)

        fake_symbol_result = pd.DataFrame(
            {
                "strategy_return": [0.01, -0.005],
                "equity_curve": [1.01, 1.00495],
            },
            index=pd.date_range("2020-01-05", periods=2, freq="D"),
        )
        fake_portfolio_result = pd.DataFrame(
            {
                "strategy_return": [0.005, 0.002],
                "equity_curve": [1.005, 1.00701],
            },
            index=pd.date_range("2020-01-05", periods=2, freq="D"),
        )

        fake_metrics_symbol = {"net_pnl": 0.00495}
        fake_metrics_portfolio = {"net_pnl": 0.00701}

        def _metrics_side_effect(df):
            if df is fake_portfolio_result:
                return fake_metrics_portfolio
            return fake_metrics_symbol

        with patch("quanttradeai.main.DataLoader.fetch_data", return_value={"AAA": _mock_data()}), \
            patch("quanttradeai.main.DataProcessor", FakeProcessor), \
            patch("quanttradeai.main.MomentumClassifier", FakeClassifier), \
            patch(
                "quanttradeai.main.simulate_trades",
                return_value={
                    "AAA": fake_symbol_result,
                    "portfolio": fake_portfolio_result,
                },
            ) as mock_sim, \
            patch(
                "quanttradeai.main.compute_metrics",
                side_effect=_metrics_side_effect,
            ) as mock_metrics:

            summary = run_model_backtest(
                model_config=model_cfg,
                model_path=model_dir,
                backtest_config=None,
            )

        mock_sim.assert_called_once()
        kwargs = mock_sim.call_args.kwargs
        assert isinstance(kwargs["portfolio"], PortfolioManager)
        assert kwargs["portfolio"].initial_capital == 100_000.0
        metric_args = [call.args[0] for call in mock_metrics.call_args_list]
        assert any(arg is fake_symbol_result for arg in metric_args)
        assert any(arg is fake_portfolio_result for arg in metric_args)
        assert summary["AAA"]["metrics"] == fake_metrics_symbol
        assert summary["portfolio"]["metrics"] == fake_metrics_portfolio
        assert summary["AAA"]["output_dir"].endswith("/AAA")
        assert summary["portfolio"]["output_dir"].endswith("/portfolio")


def test_run_model_backtest_missing_features_sets_error():
    from quanttradeai.main import run_model_backtest

    class MissingFeatureClassifier(FakeClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.feature_columns = ["f1", "f3"]

    with tempfile.TemporaryDirectory() as tmpdir:
        model_cfg = _make_config(tmpdir)
        model_dir = os.path.join(tmpdir, "fake_model")
        os.makedirs(model_dir, exist_ok=True)

        with patch("quanttradeai.main.DataLoader.fetch_data", return_value={"AAA": _mock_data()}), \
            patch("quanttradeai.main.DataProcessor", FakeProcessor), \
            patch("quanttradeai.main.MomentumClassifier", MissingFeatureClassifier), \
            patch("quanttradeai.main.simulate_trades") as mock_sim:

            summary = run_model_backtest(
                model_config=model_cfg,
                model_path=model_dir,
                backtest_config=None,
            )

        mock_sim.assert_not_called()
        assert "AAA" in summary
        assert "missing required features" in summary["AAA"]["error"].lower()
        assert "portfolio" not in summary

