import pandas as pd
import numpy as np
from quantreo.features_engineering.candle.position import internal_bar_strength


def test_internal_bar_strength(ohlcv_sample):
    """
    Test the `internal_bar_strength` function.
    """

    # === Data Preparation ===
    df = ohlcv_sample.copy().head(100)
    result = internal_bar_strength(df=df, high_col="high", low_col="low", close_col="close")

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "IBS"

    # === Value Checks ===
    assert not result.isna().all()  # not fully NaN
    assert not np.isinf(result.dropna()).any()
    assert ((result >= -0.1) & (result <= 1.1)).all(), "IBS should stay close to [0, 1]"

    # === Robustness Checks ===
    # Case where high == low → division by zero should not create Inf or NaN
    df_flat = df.copy()
    df_flat["high"] = df_flat["low"] = df_flat["close"]
    result_flat = internal_bar_strength(df_flat)
    assert not np.isinf(result_flat).any()
    assert not result_flat.isna().any()

    # === Side Effect Check ===
    df_original = ohlcv_sample.copy().head(100)
    df_copy = df_original.copy()
    internal_bar_strength(df_original)
    pd.testing.assert_frame_equal(df_original, df_copy)
