import pandas as pd
import numpy as np
import pytest
from quantreo.features_engineering.market_regime.trendiness import kama_market_regime


def test_kama_market_regime(ohlcv_sample):
    """
    Test the `kama_market_regime` function.
    """

    # === Data Preparation ===
    df = ohlcv_sample.copy().head(200)
    result = kama_market_regime(df=df, col="close")

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert set(result.unique()).issubset({1, -1})

    # === Value Checks ===
    assert not result.isna().all()
    assert not np.isinf(result).any()

    # === Robustness Checks ===
    # Missing column
    df_missing = df.drop(columns=["close"])
    with pytest.raises(ValueError, match="The required column 'close' is not present"):
        kama_market_regime(df_missing, col="close")

    # Custom parameters shouldn't break
    result_custom = kama_market_regime(
        df=df, col="close", l1_fast=30, l2_fast=2, l3_fast=10, l1_slow=100, l2_slow=2, l3_slow=20
    )
    assert isinstance(result_custom, pd.Series)
    assert set(result_custom.unique()).issubset({1, -1})

    # === Side Effect Check ===
    df_original = ohlcv_sample.copy().head(200)
    df_copy = df_original.copy()
    kama_market_regime(df_original, col="close")
    pd.testing.assert_frame_equal(df_original, df_copy)