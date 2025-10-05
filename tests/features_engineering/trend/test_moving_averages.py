import pytest
import numpy as np
import pandas as pd

from quantreo.features_engineering.trend.moving_averages import sma, kama


def test_sma(ohlcv_sample):
    """Test the sma function."""
    df = ohlcv_sample.copy().head(200)

    result = sma(df=df, col="close", window_size=30)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "sma_30"

    # === Value Checks ===
    assert result.iloc[:29].isna().all()
    valid = result.iloc[29:]
    assert not valid.isna().any()
    assert not np.isinf(valid).any()

    manual_check = df["close"].rolling(30).mean()
    pd.testing.assert_series_equal(result, manual_check, check_names=False)

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        sma(df, col="not_a_col")

    # === Side Effect Check ===
    df_original = df.copy()
    sma(df_original, col="close")
    pd.testing.assert_frame_equal(df, df_original)



def test_kama(ohlcv_sample):
    """Test the kama function."""
    df = ohlcv_sample.copy().head(200)

    result = kama(df=df, col="close", l1=10, l2=2, l3=30)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "kama"

    # === Value Checks ===
    # Le premier élément doit être égal (ou proche) du premier close
    assert np.isclose(result.iloc[0], df["close"].iloc[0], rtol=1e-6)
    valid = result.dropna()
    assert not valid.isna().any()
    assert not np.isinf(valid).any()
    assert valid.between(df["close"].min() * 0.5, df["close"].max() * 1.5).all()

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        kama(df, col="not_a_col")

    # === Side Effect Check ===
    df_original = df.copy()
    kama(df_original, col="close")
    pd.testing.assert_frame_equal(df, df_original)

    # === Logical Check ===
    diff_price = df["close"].diff().abs().mean()
    diff_kama = result.diff().abs().mean()
    assert diff_kama < diff_price  # KAMA doit lisser les variations
