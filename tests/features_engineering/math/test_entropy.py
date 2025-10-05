import pytest
import numpy as np
import pandas as pd
from quantreo.features_engineering.math.entropy import sample_entropy, spectral_entropy, permutation_entropy, petrosian_fd


def test_sample_entropy(ohlcv_sample):
    """
    Test the `sample_entropy` function.
    """
    df = ohlcv_sample.copy().head(200)
    result = sample_entropy(df, col="close", window_size=50, order=2)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)

    # === Value Checks ===
    assert result.iloc[:49].isna().all()
    valid = result.iloc[49:]
    assert not valid.isna().any()
    assert np.all(np.isfinite(valid))

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        sample_entropy(df, col="close", window_size=5)
    with pytest.raises(ValueError):
        sample_entropy(df, col="close", order=0)
    with pytest.raises(ValueError):
        sample_entropy(df, col="missing_col")

    # === Side Effect Check ===
    df_copy = df.copy()
    sample_entropy(df_copy, col="close")
    pd.testing.assert_frame_equal(df, df_copy)


def test_spectral_entropy(ohlcv_sample):
    """
    Test the `spectral_entropy` function.
    """
    df = ohlcv_sample.copy().head(200)
    result = spectral_entropy(df, col="close", window_size=60, sf=2, method="welch")

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)

    # === Value Checks ===
    assert result.iloc[:59].isna().all()
    valid = result.iloc[59:]
    assert not valid.isna().any()
    assert np.all(np.isfinite(valid))

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        spectral_entropy(df, col="close", window_size=10)
    with pytest.raises(ValueError):
        spectral_entropy(df, col="close", sf=0)
    with pytest.raises(ValueError):
        spectral_entropy(df, col="close", method="invalid")
    with pytest.raises(ValueError):
        spectral_entropy(df, col="missing_col")

    # === Side Effect Check ===
    df_copy = df.copy()
    spectral_entropy(df_copy, col="close")
    pd.testing.assert_frame_equal(df, df_copy)


def test_permutation_entropy(ohlcv_sample):
    """
    Test the `permutation_entropy` function.
    """
    df = ohlcv_sample.copy().head(200)
    result = permutation_entropy(df, col="close", window_size=50, order=3, delay=1)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)

    # === Value Checks ===
    assert result.iloc[:49].isna().all()
    valid = result.iloc[49:]
    assert not valid.isna().any()
    assert np.all(np.isfinite(valid))

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        permutation_entropy(df, col="close", window_size=5)
    with pytest.raises(ValueError):
        permutation_entropy(df, col="close", order=1)
    with pytest.raises(ValueError):
        permutation_entropy(df, col="close", delay=0)
    with pytest.raises(ValueError):
        permutation_entropy(df, col="missing_col")

    # === Side Effect Check ===
    df_copy = df.copy()
    permutation_entropy(df_copy, col="close")
    pd.testing.assert_frame_equal(df, df_copy)


def test_petrosian_fd(ohlcv_sample):
    """
    Test the `petrosian_fd` function.
    """
    df = ohlcv_sample.copy().head(200)
    result = petrosian_fd(df, col="close", window_size=50)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)

    # === Value Checks ===
    assert result.iloc[:49].isna().all()
    valid = result.iloc[49:]
    assert not valid.isna().any()
    assert np.all(np.isfinite(valid))

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        petrosian_fd(df, col="close", window_size=5)
    with pytest.raises(ValueError):
        petrosian_fd(df, col="missing_col")

    # === Side Effect Check ===
    df_copy = df.copy()
    petrosian_fd(df_copy, col="close")
    pd.testing.assert_frame_equal(df, df_copy)