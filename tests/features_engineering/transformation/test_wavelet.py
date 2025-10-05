import pytest
import numpy as np
import pandas as pd
from quantreo.features_engineering.transformation.wavelet import wavelet_transform


def test_wavelet_transform_basic_structure(ohlcv_sample):
    """
    Test the `wavelet_transform` function for structure, validity, and stability.
    """
    df = ohlcv_sample.copy().head(600)
    window_size = 128

    result = wavelet_transform(
        df,
        col="close",
        window_size=window_size,
        wavelet="db4",
        level=3,
        keep="approx",
    )

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "close_wavelet_approx_L3_db4"

    # === Value Checks ===
    # First (window_size - 1) values must be NaN
    assert result.iloc[:window_size - 1].isna().all()

    valid = result.iloc[window_size - 1:].dropna()
    assert np.all(np.isfinite(valid))
    # Wavelet smoothing should reduce volatility
    orig_std = df["close"].iloc[window_size - 1:].std()
    filt_std = valid.std()
    assert filt_std < orig_std  # smoothing property

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        wavelet_transform(df, col="missing_col")
    with pytest.raises(ValueError):
        wavelet_transform(df, col="close", window_size=8, wavelet="db10")  # too small

    # === Side Effect Check ===
    df_copy = df.copy()
    wavelet_transform(df_copy, col="close", window_size=window_size)
    pd.testing.assert_frame_equal(df, df_copy)