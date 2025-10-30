import pandas as pd
import yaml

from quanttradeai.data.processor import DataProcessor


def _write_config(tmp_path, enabled=True, operations=None):
    config = {
        "pipeline": {"steps": ["generate_multi_timeframe_features"]},
        "multi_timeframe_features": {
            "enabled": enabled,
            "operations": operations or [],
        },
    }
    config_path = tmp_path / "features.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle)
    return config_path


def test_generate_multi_timeframe_features_creates_expected_columns(tmp_path):
    operations = [
        {"type": "ratio", "timeframe": "1h", "base": "close"},
        {
            "type": "pct_change",
            "timeframe": "1h",
            "base": "close",
            "stat": "last",
            "feature_name": "mtf_pct_close_1h",
        },
        {
            "type": "rolling_divergence",
            "timeframe": "30m",
            "base": "volume",
            "stat": "sum",
            "rolling_window": 3,
        },
    ]
    config_path = _write_config(tmp_path, operations=operations)
    processor = DataProcessor(str(config_path))

    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0, 103.0],
            "Volume": [1_000, 1_050, 980, 1_020],
            "close_1h_last": [99.5, 100.5, 101.5, 102.5],
            "volume_30m_sum": [400, 500, 450, 470],
        }
    )

    result = processor._generate_multi_timeframe_features(df.copy())

    ratio_col = "mtf_ratio_close_1h_last"
    assert ratio_col in result.columns
    pd.testing.assert_series_equal(
        result[ratio_col], df["close_1h_last"] / df["Close"], check_names=False
    )

    pct_col = "mtf_pct_close_1h"
    assert pct_col in result.columns
    pd.testing.assert_series_equal(
        result[pct_col],
        (df["close_1h_last"] - df["Close"]) / df["Close"],
        check_names=False,
    )

    divergence_col = "mtf_rolling_divergence_volume_30m_sum_3"
    assert divergence_col in result.columns
    expected_ratio = df["volume_30m_sum"] / df["Volume"]
    expected = expected_ratio - expected_ratio.rolling(3).mean()
    pd.testing.assert_series_equal(
        result[divergence_col], expected, check_names=False
    )


def test_generate_multi_timeframe_features_missing_secondary_is_safe(tmp_path):
    operations = [
        {"type": "ratio", "timeframe": "2h", "base": "close", "stat": "last"}
    ]
    config_path = _write_config(tmp_path, operations=operations)
    processor = DataProcessor(str(config_path))

    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0],
            "close_1h_last": [100.0, 100.5, 101.0],
        }
    )

    result = processor._generate_multi_timeframe_features(df.copy())
    assert "mtf_ratio_close_2h_last" not in result.columns


def test_multi_timeframe_step_skipped_when_disabled(tmp_path):
    config_path = _write_config(tmp_path, enabled=False, operations=[
        {"type": "ratio", "timeframe": "1h", "base": "close"}
    ])
    processor = DataProcessor(str(config_path))

    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "close_1h_last": [100.0, 101.0],
        }
    )

    result = processor._generate_multi_timeframe_features(df.copy())
    assert list(result.columns) == ["Close", "close_1h_last"]
