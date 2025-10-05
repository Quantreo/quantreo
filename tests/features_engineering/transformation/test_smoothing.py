import pytest
import numpy as np
import pandas as pd
from quantreo.features_engineering.transformation.smoothing import mma


def test_mma_basic_structure(ohlcv_sample):
    """
    Test the `mma` (Median Moving Average) function for structure and correctness.
    """
    df = ohlcv_sample.copy().head(100)
    window_size = 5
    result = mma(df, col="close", window_size=window_size)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == f"mma_{window_size}"

    # === Value Checks ===
    # The first (window_size - 1) values must be NaN
    assert result.iloc[:window_size - 1].isna().all()

    # Remaining values should be finite and close to rolling median
    valid = result.dropna()
    expected = df["close"].rolling(window=window_size).median().dropna()
    pd.testing.assert_series_equal(valid, expected, check_names=False)

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        mma(df, col="missing_col", window_size=window_size)

    # === Side Effect Check ===
    df_copy = df.copy()
    mma(df_copy, col="close", window_size=window_size)
    pd.testing.assert_frame_equal(df, df_copy)
