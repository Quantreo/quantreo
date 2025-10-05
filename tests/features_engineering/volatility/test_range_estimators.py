import pytest
import numpy as np
import pandas as pd

from quantreo.features_engineering.volatility.range_estimators import (
    rogers_satchell_volatility,
    parkinson_volatility,
    yang_zhang_volatility,
)


def test_rogers_satchell_volatility(ohlcv_sample):
    """Test the rogers_satchell_volatility function."""
    df = ohlcv_sample.copy().head(300)

    result = rogers_satchell_volatility(df=df, window_size=30)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "rogers_satchell_vol"

    # === Value Checks ===
    assert result.iloc[:30].isna().all()
    valid = result.iloc[30:]
    assert not valid.isna().any()
    assert not np.isinf(valid).any()
    assert (valid >= 0).all()

    # Vérifie qu'une série constante donne vol ≈ 0
    flat_df = pd.DataFrame({
        "high": np.ones(100),
        "low": np.ones(100),
        "open": np.ones(100),
        "close": np.ones(100),
    })
    flat_vol = rogers_satchell_volatility(flat_df, window_size=10)
    assert np.allclose(flat_vol.dropna(), 0.0, atol=1e-12)

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        rogers_satchell_volatility(df.drop(columns=["high"]))

    # === Side Effect Check ===
    df_original = df.copy()
    rogers_satchell_volatility(df_original)
    pd.testing.assert_frame_equal(df, df_original)


def test_parkinson_volatility(ohlcv_sample):
    """Test the parkinson_volatility function."""
    df = ohlcv_sample.copy().head(300)

    result = parkinson_volatility(df=df, window_size=30)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "rolling_volatility_vol"

    # === Value Checks ===
    assert result.iloc[:30].isna().all()
    valid = result.iloc[30:]
    assert not valid.isna().any()
    assert not np.isinf(valid).any()
    assert (valid >= 0).all()

    # Série constante → vol ≈ 0
    flat_df = pd.DataFrame({"high": np.ones(100), "low": np.ones(100)})
    flat_vol = parkinson_volatility(flat_df, window_size=10)
    assert np.allclose(flat_vol.dropna(), 0.0, atol=1e-12)

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        parkinson_volatility(df.drop(columns=["low"]))

    # === Side Effect Check ===
    df_original = df.copy()
    parkinson_volatility(df_original)
    pd.testing.assert_frame_equal(df, df_original)


def test_yang_zhang_volatility(ohlcv_sample):
    """Test the yang_zhang_volatility function."""
    df = ohlcv_sample.copy().head(300)

    result = yang_zhang_volatility(df=df, window_size=30)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "yang_zhang_vol"

    # === Value Checks ===
    assert result.iloc[:30].isna().all()
    valid = result.iloc[30:]
    assert not valid.isna().any()
    assert not np.isinf(valid).any()
    assert (valid >= 0).all()

    # Série constante → vol ≈ 0
    flat_df = pd.DataFrame({
        "high": np.ones(100),
        "low": np.ones(100),
        "open": np.ones(100),
        "close": np.ones(100),
    })
    flat_vol = yang_zhang_volatility(flat_df, window_size=10)
    assert np.allclose(flat_vol.dropna(), 0.0, atol=1e-12)

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        yang_zhang_volatility(df.drop(columns=["close"]))

    # === Side Effect Check ===
    df_original = df.copy()
    yang_zhang_volatility(df_original)
    pd.testing.assert_frame_equal(df, df_original)