import pandas as pd
import numpy as np
import pytest
from quantreo.features_engineering.math.correlation import auto_corr


def test_auto_corr(ohlcv_sample):
    """
    Test the `auto_corr` function.
    """

    # === Data Preparation ===
    df = ohlcv_sample.copy().head(200)
    result = auto_corr(df=df, col="close", window_size=50, lag=10)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "autocorr_10"

    # === NaN / Inf Checks ===
    # First (window_size - 1) values should be NaN
    assert result.iloc[:49].isna().all(), "First window should be NaN due to insufficient data"
    # After that, no NaN or Inf
    valid = result.iloc[49:]
    assert not valid.isna().any(), "No NaN allowed after the warm-up period"
    assert not np.isinf(valid).any(), "No Inf allowed in the autocorrelation output"

    # === Value Checks ===
    # Autocorrelation must be between -1 and 1
    assert ((valid >= -1) & (valid <= 1)).all(), "Autocorrelation should be between -1 and 1"

    # === Robustness Checks ===
    # Invalid column should raise a ValueError
    df_missing = df.drop(columns=["close"])
    with pytest.raises(KeyError):
        auto_corr(df_missing, col="close")

    # Different lag should produce a differently named Series
    result_lag_5 = auto_corr(df, col="close", window_size=50, lag=5)
    assert result_lag_5.name == "autocorr_5"

    # === Side Effect Check ===
    df_original = ohlcv_sample.copy().head(200)
    df_copy = df_original.copy()
    auto_corr(df_original, col="close", window_size=50, lag=10)
    pd.testing.assert_frame_equal(df_original, df_copy)
