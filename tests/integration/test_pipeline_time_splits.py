from unittest.mock import patch

import pandas as pd
import yaml

from quanttradeai.main import time_aware_split, run_pipeline


def test_time_aware_split_with_window():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": range(10)}, index=idx)
    cfg = {"data": {"test_start": "2024-01-06", "test_end": "2024-01-08"}}
    train, test = time_aware_split(df, cfg)
    assert train.index.max() < pd.to_datetime("2024-01-06")
    assert test.index.min() == pd.to_datetime("2024-01-06")
    assert test.index.max() == pd.to_datetime("2024-01-08")
    assert len(train) == 5 and len(test) == 3


def test_time_aware_split_with_start_only():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": range(10)}, index=idx)
    cfg = {"data": {"test_start": "2024-01-06"}}
    train, test = time_aware_split(df, cfg)
    assert train.index.max() < pd.to_datetime("2024-01-06")
    assert test.index.min() == pd.to_datetime("2024-01-06")
    assert len(train) == 5 and len(test) == 5


def test_time_aware_split_fallback_fraction():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": range(10)}, index=idx)
    cfg = {"training": {"test_size": 0.2}}
    train, test = time_aware_split(df, cfg)
    assert len(train) == 8 and len(test) == 2
    assert train.index.max() < test.index.min()


def test_pipeline_handles_secondary_timeframes(tmp_path):
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(10),
            "High": range(10),
            "Low": range(10),
            "Close": range(10),
            "Volume": [100] * 10,
            "close_1h_last": range(10),
            "volume_30m_sum": [200] * 10,
        },
        index=index,
    )

    config = {
        "data": {
            "symbols": ["AAA"],
            "start_date": "2020-01-01",
            "end_date": "2020-01-10",
            "timeframe": "1d",
            "secondary_timeframes": ["1h", "30m"],
            "cache_path": str(tmp_path),
            "use_cache": False,
            "refresh": False,
        },
        "training": {"test_size": 0.2},
    }

    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as fh:
        yaml.safe_dump(config, fh)

    with patch("quanttradeai.main.DataLoader") as mock_loader, patch(
        "quanttradeai.main.DataProcessor"
    ) as mock_processor, patch("quanttradeai.main.MomentumClassifier") as mock_model:
        loader_instance = mock_loader.return_value
        loader_instance.fetch_data.return_value = {"AAA": df}

        def process_passthrough(input_df):
            assert "close_1h_last" in input_df.columns
            return input_df

        processor_instance = mock_processor.return_value
        processor_instance.process_data.side_effect = process_passthrough

        def generate_labels_passthrough(input_df, **_):
            assert "volume_30m_sum" in input_df.columns
            labeled = input_df.copy()
            labeled["label"] = 0
            return labeled

        processor_instance.generate_labels.side_effect = generate_labels_passthrough

        model_instance = mock_model.return_value

        def prepare_data(df):
            features = pd.DataFrame({"feature": range(len(df))}, index=df.index)
            labels = pd.Series([0] * len(df), index=df.index)
            return features, labels

        model_instance.prepare_data.side_effect = prepare_data
        model_instance.optimize_hyperparameters.return_value = {}
        model_instance.train.return_value = None
        model_instance.evaluate.return_value = {"accuracy": 1.0}
        model_instance.save_model.return_value = None

        results = run_pipeline(str(config_path))

    mock_loader.assert_called_once_with(str(config_path))
    assert "AAA" in results
    assert "hyperparameters" in results["AAA"]

