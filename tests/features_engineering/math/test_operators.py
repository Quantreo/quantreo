import pytest
import numpy as np
import pandas as pd
from quantreo.features_engineering.math.operators import derivatives, log_pct


def test_derivatives_basic_structure(ohlcv_sample):
    """
    Test the `derivatives` function for structure and correctness.
    """
    df = ohlcv_sample.copy().head(200)
    velocity, acceleration = derivatives(df, col="close")

    # === Structural Checks ===
    assert isinstance(velocity, pd.Series)
    assert isinstance(acceleration, pd.Series)
    assert len(velocity) == len(df)
    assert len(acceleration) == len(df)
    assert velocity.index.equals(df.index)
    assert acceleration.index.equals(df.index)

    # === Value Checks ===
    assert np.all(np.isfinite(velocity))
    assert np.all(np.isfinite(acceleration))
    assert (velocity.iloc[1:] == df["close"].diff().iloc[1:]).all()
    assert (acceleration.iloc[2:] == velocity.diff().iloc[2:]).all()

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        derivatives(df, col="missing_col")

    # === Side Effect Check ===
    df_copy = df.copy()
    derivatives(df_copy, col="close")
    pd.testing.assert_frame_equal(df, df_copy)


def test_log_pct_basic_structure(ohlcv_sample):
    """
    Test the `log_pct` function for correct structure and expected values.
    """
    df = ohlcv_sample.copy().head(200)
    window_size = 10
    result = log_pct(df, col="close", window_size=window_size)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)

    # === Value Checks ===
    # NaN for first `window_size` values
    assert result.iloc[:window_size].isna().all()
    valid = result.iloc[window_size:]
    assert not valid.isna().any()
    assert np.isfinite(valid).all()

    # Mathematical correctness: log(P_t / P_{t-n})
    expected = np.log(df["close"] / df["close"].shift(window_size))
    pd.testing.assert_series_equal(result, expected, check_names=False)

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        log_pct(df, col="missing_col", window_size=window_size)

    # === Side Effect Check ===
    df_copy = df.copy()
    log_pct(df_copy, col="close", window_size=window_size)
    pd.testing.assert_frame_equal(df, df_copy)
