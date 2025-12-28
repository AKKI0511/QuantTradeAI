import json
from pathlib import Path

import pandas as pd
import pytest

from quanttradeai.data.loader import DataLoader
from quanttradeai.data.processor import DataProcessor
from quanttradeai.main import run_model_backtest, run_pipeline
from quanttradeai.models.classifier import MomentumClassifier


@pytest.fixture
def sample_config_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "config" / "model_config.yaml")


@pytest.fixture
def invalid_data() -> dict:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Missing Volume column to force validation failure
    df = pd.DataFrame(
        {"Open": 1, "High": 1, "Low": 1, "Close": 1}, index=dates
    )
    return {"AAPL": df}


@pytest.fixture
def short_data() -> dict:
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(10),
            "High": range(10),
            "Low": range(10),
            "Close": range(10),
            "Volume": 1,
        },
        index=dates,
    )
    return {"AAPL": df}


def test_run_pipeline_validation_failure_writes_report(
    tmp_path, monkeypatch, sample_config_path, invalid_data
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DataLoader, "fetch_data", lambda self, refresh=None: invalid_data)

    with pytest.raises(ValueError):
        run_pipeline(sample_config_path)

    reports = list((tmp_path / "models" / "experiments").rglob("validation.json"))
    assert reports, "Validation report was not written"

    content = json.loads(reports[0].read_text())
    assert content["AAPL"]["missing_columns"], "Report should note missing columns"
    assert content["AAPL"]["passed"] is False


def test_run_pipeline_skip_validation_allows_progression(
    tmp_path, monkeypatch, sample_config_path, short_data
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DataLoader, "fetch_data", lambda self, refresh=None: short_data)
    monkeypatch.setattr(DataProcessor, "process_data", lambda self, df: df.assign(feature=1))
    monkeypatch.setattr(
        DataProcessor, "generate_labels", lambda self, df: df.assign(label=1)
    )
    monkeypatch.setattr(
        MomentumClassifier,
        "prepare_data",
        lambda self, df: (df[[c for c in df.columns if c not in ["label"]]].values, df["label"].values),
    )
    monkeypatch.setattr(
        MomentumClassifier, "optimize_hyperparameters", lambda self, X, y, n_trials=50: {}
    )
    monkeypatch.setattr(MomentumClassifier, "train", lambda self, X, y, params=None: None)
    monkeypatch.setattr(
        MomentumClassifier, "evaluate", lambda self, X, y: {"accuracy": 1.0}
    )
    monkeypatch.setattr(MomentumClassifier, "save_model", lambda self, path: None)

    results, coverage_info = run_pipeline(sample_config_path, skip_validation=True)
    assert "AAPL" in results
    assert coverage_info["path"].endswith("test_window_coverage.json")


def test_run_model_backtest_validates_before_execution(
    tmp_path, monkeypatch, sample_config_path, invalid_data
):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(DataLoader, "fetch_data", lambda self: invalid_data)
    monkeypatch.setattr(MomentumClassifier, "load_model", lambda self, path: None)

    with pytest.raises(ValueError):
        run_model_backtest(model_config=sample_config_path, model_path=str(model_dir))

    report = tmp_path / "reports" / "backtests"
    validation_files = list(report.rglob("validation.json"))
    assert validation_files, "Backtest validation report missing"
