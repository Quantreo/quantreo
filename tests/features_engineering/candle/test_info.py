import pandas as pd
import numpy as np
from quantreo.features_engineering.candle.info import candle_information


def test_candle_information(ohlcv_sample):
    """
    Test the `candle_information` function on a known dataset.
    """

    # === Data Preparation ===
    df = ohlcv_sample.copy().head(100)
    df["candle_way"], df["filling"], df["amplitude"] = candle_information(
        df=df, open_col="open", high_col="high", low_col="low", close_col="close"
    )

    # === Structural Checks ===
    for col in ["candle_way", "filling", "amplitude"]:
        assert col in df.columns, f"{col} not created"
        assert len(df[col]) == len(df), f"{col} length mismatch"
        assert not df[col].isna().any(), f"{col} contains NaN"
        assert not np.isinf(df[col]).any(), f"{col} contains Inf"

    # === Type Checks ===
    assert df["candle_way"].dtype in [np.int32, np.int64]
    assert np.issubdtype(df["filling"].dtype, np.floating)
    assert np.issubdtype(df["amplitude"].dtype, np.floating)

    # === Range Checks ===
    assert df["filling"].between(0, 1).all()
    assert (df["amplitude"] >= 0).all()

    # === Side Effect Check ===
    df_original = ohlcv_sample.copy().head(100)
    df_copy = df_original.copy()
    candle_information(df_original)
    pd.testing.assert_frame_equal(df_original, df_copy)
