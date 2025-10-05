import pytest
import pandas as pd
import numpy as np
from quantreo.features_engineering.math.fractal import hurst, detrended_fluctuation


def test_hurst_basic_structure(ohlcv_sample):
    """
    Test the `hurst` function for correct output structure and validity.
    """
    df = ohlcv_sample.copy().head(300)
    result = hurst(df, col="close", window_size=120)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "hurst_120"

    # === Value Checks ===
    assert result.iloc[:119].isna().all()
    valid = result.iloc[119:]
    assert not valid.isna().all()  # some values may be NaN but not all
    assert np.isfinite(valid.dropna()).all()

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        hurst(df, col="missing_col", window_size=120)
    with pytest.raises(ValueError):
        hurst(df, col="close", window_size=50)

    # === Side Effect Check ===
    df_copy = df.copy()
    hurst(df_copy, col="close", window_size=120)
    pd.testing.assert_frame_equal(df, df_copy)


def test_detrended_fluctuation_basic_structure(ohlcv_sample):
    """
    Test the `detrended_fluctuation` function for correct output structure and validity.
    """
    df = ohlcv_sample.copy().head(300)
    result = detrended_fluctuation(df, col="close", window_size=150)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)

    # === Value Checks ===
    assert result.iloc[:149].isna().all()
    valid = result.iloc[149:]
    assert not valid.isna().all()
    assert np.isfinite(valid.dropna()).all()

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        detrended_fluctuation(df, col="missing_col", window_size=150)
    with pytest.raises(ValueError):
        detrended_fluctuation(df, col="close", window_size=50)

    # === Side Effect Check ===
    df_copy = df.copy()
    detrended_fluctuation(df_copy, col="close", window_size=150)
    pd.testing.assert_frame_equal(df, df_copy)
