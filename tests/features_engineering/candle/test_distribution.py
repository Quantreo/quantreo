import pandas as pd
import numpy as np
from quantreo.features_engineering.candle.distribution import price_distribution


def test_price_distribution(ohlcv_sample):
    """
    Test the `price_distribution` function.
    """

    # === Data Preparation ===
    df = ohlcv_sample.copy().head(200)
    result = price_distribution(df=df, col="close", window_size=50, start_percentage=0.25, end_percentage=0.75)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)

    # === Value Checks ===
    assert result.iloc[:49].isna().all()  # first (window_size - 1) should be NaN
    valid = result.iloc[49:]
    assert not valid.isna().any()
    assert (valid >= 0).all() and (valid <= 100).all()

    # === Robustness Checks ===
    df_nan = df.copy()
    df_nan.loc[df_nan.index[:10], "close"] = np.nan
    result_nan = price_distribution(df_nan, col="close", window_size=50)
    assert len(result_nan) == len(df_nan)
    assert not np.isinf(result_nan.dropna()).any()

    # === Side Effect Check ===
    df_original = df.copy()
    df_copy = df.copy()
    price_distribution(df_original, col="close", window_size=50)
    pd.testing.assert_frame_equal(df_original, df_copy)
