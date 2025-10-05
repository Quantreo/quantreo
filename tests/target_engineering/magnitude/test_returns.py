import numpy as np
import pandas as pd
import pytest
from quantreo.target_engineering.magnitude.returns import future_returns


def test_future_returns(ohlcv_sample):
    """Test the future_returns function."""
    df = ohlcv_sample.copy().head(300)

    # === Basic functional call ===
    result = future_returns(df, close_col="close", window_size=10, log_return=True)

    # === Structural Checks ===
    # Ensure the function returns a Series aligned with the DataFrame
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "fut_ret" or result.name is None  # depending on implementation

    # === Value Checks ===
    # NaNs should exist at the end due to forward shifting
    assert result.iloc[-10:].isna().all()

    # Remaining values should be finite
    valid = result.iloc[:-10]
    assert not valid.isna().any()
    assert np.isfinite(valid).all()

    # Log-return version: verify calculation consistency with manual computation
    manual_log = np.log(df["close"].shift(-10)) - np.log(df["close"])
    pd.testing.assert_series_equal(result, manual_log, check_names=False)

    # === Alternative Mode: Simple Return ===
    result_simple = future_returns(df, close_col="close", window_size=10, log_return=False)
    manual_simple = df["close"].shift(-10) / df["close"] - 1
    pd.testing.assert_series_equal(result_simple, manual_simple, check_names=False)

    # === Robustness Checks ===
    # Missing column should raise ValueError
    bad_df = df.drop(columns=["close"])
    with pytest.raises(ValueError):
        future_returns(bad_df)

    # window_size too large should produce all NaN (no error)
    big_result = future_returns(df, window_size=len(df) + 1)
    assert big_result.isna().all()

    # === Side Effect Check ===
    # Ensure the original DataFrame remains unchanged after processing
    df_original = df.copy()
    future_returns(df_original, close_col="close", window_size=10)
    pd.testing.assert_frame_equal(df, df_original)
