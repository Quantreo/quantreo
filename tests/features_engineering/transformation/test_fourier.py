import pytest
import numpy as np
import pandas as pd
from quantreo.features_engineering.transformation.fourier import fourier_transform


def test_fourier_transform_basic_structure(ohlcv_sample):
    """
    Test the `fourier_transform` function for structure, alignment, and stability.
    """
    df = ohlcv_sample.copy().head(600)
    window_size = 128

    result = fourier_transform(
        df,
        col="close",
        window_size=window_size,
        mode="topk",
        top_k=5,
        keep_dc=True,
    )

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "close_fft_topk_k5"

    # === Value Checks ===
    assert result.iloc[:window_size - 1].isna().all()
    valid = result.iloc[window_size - 1:].dropna()
    assert np.all(np.isfinite(valid))
    # Should be roughly on the same order of magnitude as the original
    assert np.isclose(valid.mean(), df["close"].mean(), rtol=0.5)

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        fourier_transform(df, col="missing_col")

    # === Side Effect Check ===
    df_copy = df.copy()
    fourier_transform(df_copy, col="close", window_size=window_size)
    pd.testing.assert_frame_equal(df, df_copy)


def test_fourier_transform_lowpass_mode(ohlcv_sample):
    """
    Ensure that 'lowpass' mode with fmax_ratio works and produces valid results.
    """
    df = ohlcv_sample.copy().head(400)
    result = fourier_transform(
        df,
        col="close",
        window_size=128,
        mode="lowpass",
        fmax_ratio=0.3,
        keep_dc=False,
    )

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.name == "close_fft_lp_0.3"

    # === Value Checks ===
    valid = result.dropna()
    assert np.all(np.isfinite(valid))
    # Mean should be near zero since DC removed
    assert abs(valid.mean()) < abs(df["close"].mean())