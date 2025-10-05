import pandas as pd
import numpy as np
import pytest
from quantreo.features_engineering.math.distribution import skewness, kurtosis, tail_index, bimodality_coefficient


def test_skewness(ohlcv_sample):
    """
    Test the `skewness` function.
    """

    # === Data Preparation ===
    df = ohlcv_sample.copy().head(200)
    result = skewness(df=df, col="close", window_size=60)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "skewness"

    # === Value Checks ===
    assert result.iloc[:59].isna().all()
    valid = result.iloc[59:]
    assert not valid.isna().any()
    assert not np.isinf(valid).any()

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        skewness(df, col="missing_col")

    with pytest.raises(ValueError):
        skewness(df.astype(str), col="close")

    with pytest.raises(ValueError):
        skewness(df, col="close", window_size=0)


def test_kurtosis(ohlcv_sample):
    """
    Test the `kurtosis` function.
    """

    # === Data Preparation ===
    df = ohlcv_sample.copy().head(200)
    result = kurtosis(df=df, col="close", window_size=60)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "kurtosis"

    # === NaN / Inf Checks ===
    assert result.iloc[:59].isna().all()
    valid = result.iloc[59:]
    assert not valid.isna().any()
    assert not np.isinf(valid).any()

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        kurtosis(df, col="missing_col")

    with pytest.raises(ValueError):
        kurtosis(df.astype(str), col="close")

    with pytest.raises(ValueError):
        kurtosis(df, col="close", window_size=-5)


def test_bimodality_coefficient(ohlcv_sample):
    """
    Test the `bimodality_coefficient` function.
    """

    # === Data Preparation ===
    df = ohlcv_sample.copy().head(200)
    result = bimodality_coefficient(df=df, col="close", window_size=100)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "bimodality_coefficient_100"

    # === NaN / Inf Checks ===
    assert result.iloc[:99].isna().all()
    valid = result.iloc[99:]
    assert not np.isinf(valid.dropna()).any()

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        bimodality_coefficient(df, col="missing_col")

    with pytest.raises(ValueError):
        bimodality_coefficient(df.astype(str), col="close")

    with pytest.raises(ValueError):
        bimodality_coefficient(df, col="close", window_size=20)


def test_tail_index(ohlcv_sample):
    """
    Test the `tail_index` function.
    """

    # === Data Preparation ===
    df = ohlcv_sample.copy().head(300)
    df["abs_close"] = df["close"].abs() + 1e-3
    result = tail_index(df=df, col="abs_close", window_size=100, k_ratio=0.1)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "hill_abs_close"

    # === Value Checks ===
    assert result.iloc[:99].isna().all()
    valid = result.iloc[99:]
    assert not np.isinf(valid.dropna()).any()

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        tail_index(df, col="abs_close", window_size=100, k_ratio=-0.1)

    with pytest.raises(ValueError):
        tail_index(df, col="abs_close", window_size=100, k_ratio=1.0)

    with pytest.raises(KeyError):
        tail_index(df, col="missing_col")