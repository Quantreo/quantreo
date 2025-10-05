import numpy as np
import pandas as pd
import pytest
from quantreo.target_engineering.magnitude.volatility import future_volatility
from quantreo.features_engineering.volatility import (
    close_to_close_volatility,
    parkinson_volatility,
    rogers_satchell_volatility,
    yang_zhang_volatility,
)


def test_future_volatility(ohlcv_sample):
    """Test the future_volatility function with multiple methods."""
    df = ohlcv_sample.copy().head(300)
    window = 20

    # === Basic functional call (close_to_close) ===
    vol = future_volatility(df, method="close_to_close", window_size=window, shift_forward=True)

    # === Structural Checks ===
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(df)
    assert vol.index.equals(df.index)
    assert vol.name == "future_volatility"

    # === Value Checks ===
    # NaNs should exist at both start and end (due to rolling + shifting)
    assert vol.isna().sum() > 0
    valid = vol.dropna()
    assert np.isfinite(valid).all()

    # Compare shape and general magnitude with direct volatility estimation
    direct = close_to_close_volatility(df, window_size=window)
    assert np.allclose(valid.mean(), direct.mean(), rtol=0.5)

    # === Test alternative volatility estimators ===
    for method in ["parkinson", "rogers_satchell", "yang_zhang"]:
        result = future_volatility(df, method=method, window_size=window)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert result.name == "future_volatility"
        assert np.isfinite(result.dropna()).all()

    # === Robustness Checks ===
    # Invalid method name should raise ValueError
    with pytest.raises(ValueError):
        future_volatility(df, method="invalid_method")

    # window_size too large should yield all NaN (not an error)
    big_result = future_volatility(df, window_size=len(df) + 5)
    assert big_result.isna().all()

    # === Shift alignment test ===
    # When shift_forward=False, volatility should not be shifted
    no_shift = future_volatility(df, window_size=window, shift_forward=False)
    assert not no_shift.equals(vol)

    # === Side Effect Check ===
    # Ensure original DataFrame is not modified
    df_original = df.copy()
    future_volatility(df_original, method="close_to_close", window_size=window)
    pd.testing.assert_frame_equal(df, df_original)
