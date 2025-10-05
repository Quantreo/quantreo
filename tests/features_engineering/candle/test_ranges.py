import pandas as pd
import numpy as np
import pytest
from quantreo.features_engineering.candle.ranges import compute_spread


def test_compute_spread(ohlcv_sample):
    """
    Test the `compute_spread` function.
    """

    # === Data Preparation ===
    df = ohlcv_sample.copy().head(100)
    result = compute_spread(df=df, high_col="high", low_col="low")

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "spread"

    # === Value Checks ===
    assert not result.isna().any()
    assert not np.isinf(result).any()
    assert (result >= 0).all(), "Spread should always be positive or zero"

    # === Robustness Checks ===
    # Case with missing column
    df_missing = df.drop(columns=["high"])
    with pytest.raises(ValueError, match="The required column 'high' is not present"):
        compute_spread(df_missing)

    # Case with high == low → spread should be 0
    df_flat = df.copy()
    df_flat["high"] = df_flat["low"]
    result_flat = compute_spread(df_flat)
    assert (result_flat == 0).all()

    # === Side Effect Check ===
    df_original = ohlcv_sample.copy().head(100)
    df_copy = df_original.copy()
    compute_spread(df_original)
    pd.testing.assert_frame_equal(df_original, df_copy)