import pytest
import pandas as pd
from quantreo.datasets import load_generated_ohlcv, load_generated_ticks, load_generated_ohlcv_with_time


@pytest.fixture
def ohlcv_sample():
    return load_generated_ohlcv().loc["2016"]


@pytest.fixture
def ticks_sample():
    df = load_generated_ticks().copy()

    assert isinstance(df.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
    assert set(["price", "volume"]).issubset(df.columns), "Missing expected columns"

    return df.head(10_000)


@pytest.fixture
def ohlcv_with_time_sample():
    df = load_generated_ohlcv_with_time().copy()

    assert isinstance(df.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
    expected_cols = ["open", "high", "low", "close", "volume", "high_time", "low_time"]
    assert all(col in df.columns for col in expected_cols), "Missing expected columns"

    return df.loc["2016"]