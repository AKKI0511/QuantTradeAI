import pandas as pd
from quanttradeai.main import time_aware_split


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

