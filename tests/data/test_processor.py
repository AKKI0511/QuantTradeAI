from pathlib import Path

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml


def load_processor_module():
    spec = importlib.util.spec_from_file_location(
        "quanttradeai.data.processor", Path("quanttradeai/data/processor.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load quanttradeai.data.processor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def load_data_processor():
    return load_processor_module().DataProcessor


def write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "features_config.yaml"
    with path.open("w") as f:
        yaml.safe_dump(config, f)
    return path


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0],
            "High": [11.0, 12.0, 13.5, 14.0],
            "Low": [9.5, 10.0, 11.0, 12.0],
            "Close": [10.5, 11.5, 12.5, 13.0],
            "Volume": [100, 110, 120, 130],
        }
    )


def test_price_features_from_yaml(tmp_path: Path):
    config = {
        "pipeline": {"steps": []},
        "price_features": ["close_to_open", "price_range"],
    }
    config_path = write_config(tmp_path, config)
    processor = load_data_processor()(config_path=str(config_path))

    df = sample_df()
    result = processor._add_price_features(df.copy())

    expected_close_to_open = (df["Close"] - df["Open"]) / df["Open"]
    expected_price_range = (df["High"] - df["Low"]) / df["Close"]

    pd.testing.assert_series_equal(
        result["close_to_open"], expected_close_to_open, check_names=False
    )
    pd.testing.assert_series_equal(
        result["price_range"], expected_price_range, check_names=False
    )


def test_volume_features_respect_periods(tmp_path: Path):
    config = {
        "pipeline": {"steps": []},
        "volume_features": [
            {"volume_sma": {"periods": [2]}},
            {"volume_sma_ratios": [2]},
            "on_balance_volume",
            "volume_price_trend",
        ],
    }
    config_path = write_config(tmp_path, config)
    processor = load_data_processor()(config_path=str(config_path))

    df = sample_df()
    result = processor._add_volume_features(df.copy())

    assert "volume_sma_2" in result.columns
    assert "volume_sma_ratio_2" in result.columns
    assert "volume_ema_5" not in result.columns  # default EMA not requested
    assert "obv" in result.columns
    assert "volume_price_trend" in result.columns


def test_volatility_and_custom_features_from_yaml(tmp_path: Path):
    config = {
        "pipeline": {"steps": []},
        "volatility_features": [
            {"atr_periods": [2]},
            {"keltner_channels": {"periods": [2], "atr_multiple": 1.5}},
            {"bollinger_bands": {"period": 3, "std_dev": 1}},
        ],
        "custom_features": [
            {"price_momentum": [2]},
            {"volume_momentum": [2]},
            {"mean_reversion": [2]},
            {"volatility_breakout": {"lookback": [2], "threshold": 1.5}},
        ],
    }
    config_path = write_config(tmp_path, config)
    processor = load_data_processor()(config_path=str(config_path))

    df = sample_df()
    vol_result = processor._add_volatility_features(df.copy())
    assert "atr_2" in vol_result.columns
    assert "keltner_upper_2" in vol_result.columns
    assert "bb_upper" in vol_result.columns

    custom_result = processor._add_custom_features(vol_result.copy())
    assert "price_momentum_2" in custom_result.columns
    assert "volume_momentum_2" in custom_result.columns
    assert "mean_reversion_2" in custom_result.columns
    assert "volatility_breakout_2" in custom_result.columns


def test_feature_preprocessor_uses_train_statistics_only():
    processor_module = load_processor_module()
    preprocessor = processor_module.FeaturePreprocessor(
        scaling_method="standard",
        outlier_method="winsorize",
        outlier_limits=[0.25, 0.75],
        excluded_columns={"Open", "High", "Low", "Close", "Volume", "label"},
    )

    train_df = pd.DataFrame(
        {
            "Open": [1.0, 1.0, 1.0, 1.0],
            "High": [1.0, 1.0, 1.0, 1.0],
            "Low": [1.0, 1.0, 1.0, 1.0],
            "Close": [1.0, 1.0, 1.0, 1.0],
            "Volume": [10.0, 10.0, 10.0, 10.0],
            "alpha_feature": [0.0, 1.0, 2.0, 3.0],
            "label": [0, 0, 0, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.0],
            "Low": [1.0],
            "Close": [1.0],
            "Volume": [10.0],
            "alpha_feature": [100.0],
            "label": [0],
        }
    )

    transformed_train = preprocessor.fit_transform(train_df)
    transformed_test = preprocessor.transform(test_df)

    train_clipped = train_df["alpha_feature"].clip(0.75, 2.25)
    expected_test = (2.25 - train_clipped.mean()) / train_clipped.std(ddof=0)

    assert preprocessor.clip_bounds["alpha_feature"]["upper"] == pytest.approx(2.25)
    assert transformed_train["alpha_feature"].mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed_test["alpha_feature"].iloc[0] == pytest.approx(expected_test)
    assert transformed_test["Close"].iloc[0] == pytest.approx(1.0)

