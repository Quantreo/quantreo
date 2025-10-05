import pytest
import numpy as np
import pandas as pd
from quantreo.features_engineering.transformation.filters import savgol_filter


def test_savgol_filter_structure_and_values(ohlcv_sample):
    """
    Test the `savgol_filter` function for structure, shape, and value integrity.
    """
    df = ohlcv_sample.copy().head(300)
    result = savgol_filter(df, col="close", window_size=11, polyorder=3)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "close_savgol_causal"

    # === Value Checks ===
    # first (window_size - 1) should be NaN (no causal window yet)
    assert result.iloc[:10].isna().all()
    valid = result.iloc[10:]
    assert not valid.isna().any()
    assert np.isfinite(valid).all()

    # The smoothed series should not differ wildly from the original
    diff = np.abs(valid - df["close"].iloc[10:])
    assert diff.mean() < df["close"].std()  # smoothness check

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        savgol_filter(df, col="close", window_size=10, polyorder=3)  # even window
    with pytest.raises(ValueError):
        savgol_filter(df, col="missing_col", window_size=11, polyorder=3)  # missing column
    with pytest.raises(ValueError):
        savgol_filter(df, col="close", window_size=7, polyorder=7)  # invalid polyorder

    # === Side Effect Check ===
    df_copy = df.copy()
    savgol_filter(df_copy, col="close", window_size=11, polyorder=3)
    pd.testing.assert_frame_equal(df, df_copy)


def test_savgol_filter_monotonic_input():
    """
    Ensure `savgol_filter` behaves correctly on a simple monotonic series (linear trend).
    """
    n = 50
    df = pd.DataFrame({"close": np.linspace(1, 100, n)}, index=pd.RangeIndex(n))
    result = savgol_filter(df, col="close", window_size=9, polyorder=2)

    # The smoothed result should closely follow the input since it's a perfect polynomial
    diff = np.abs(result.dropna() - df["close"].iloc[8:])
    assert diff.mean() < 1e-6  # almost exact for a quadratic polynomial


def test_savgol_filter_noise_reduction():
    """
    Test that `savgol_filter` reduces noise in a random series (variance reduction).
    """
    np.random.seed(42)
    noise = np.random.normal(0, 1, 200)
    signal = np.linspace(0, 10, 200)
    df = pd.DataFrame({"close": signal + noise})
    result = savgol_filter(df, col="close", window_size=11, polyorder=3)

    original_var = df["close"].var()
    filtered_var = result.dropna().var()

    assert filtered_var < original_var  # should reduce variance (noise smoothing)
