import pytest
import numpy as np
import pandas as pd
from quantreo.features_engineering.math.statistical_tests import adf_test, arch_test, shapiro_wilk

def test_adf_test_structure_and_validity(ohlcv_sample):
    """
    Test the `adf_test` function for correct structure, rolling behavior, and validity.
    """
    df = ohlcv_sample.copy().head(300)
    window_size = 60

    adf_stat, adf_pval = adf_test(df, col="close", window_size=window_size, lags=2, regression="c")

    # === Structural Checks ===
    assert isinstance(adf_stat, pd.Series)
    assert isinstance(adf_pval, pd.Series)
    assert len(adf_stat) == len(df)
    assert len(adf_pval) == len(df)
    assert adf_stat.index.equals(df.index)
    assert adf_pval.index.equals(df.index)
    assert adf_stat.name == "adf_stat"
    assert adf_pval.name == "adf_pval"

    # === Value Checks ===
    assert adf_stat.iloc[:window_size - 1].isna().all()
    assert adf_pval.iloc[:window_size - 1].isna().all()

    valid_stat = adf_stat.iloc[window_size - 1:].dropna()
    valid_pval = adf_pval.iloc[window_size - 1:].dropna()

    assert np.all(np.isfinite(valid_stat))
    assert np.all((valid_pval >= 0) & (valid_pval <= 1))

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        adf_test(df, col="missing_col", window_size=window_size)
    with pytest.raises(ValueError):
        adf_test(df.assign(close="not_numeric"), col="close", window_size=window_size)
    with pytest.raises(ValueError):
        adf_test(df, col="close", window_size=5)  # window too small
    with pytest.raises(ValueError):
        adf_test(df, col="close", window_size=window_size, lags=-1)
    with pytest.raises(ValueError):
        adf_test(df, col="close", window_size=window_size, regression="invalid")

    # === Side Effect Check ===
    df_copy = df.copy()
    adf_test(df_copy, col="close", window_size=window_size)
    pd.testing.assert_frame_equal(df, df_copy)


def test_adf_test_regression_ct(ohlcv_sample):
    """
    Ensure the `adf_test` function runs properly with regression='ct'.
    """
    df = ohlcv_sample.copy().head(250)
    adf_stat, adf_pval = adf_test(df, col="close", window_size=50, regression="ct")

    # Ensure proper types and length
    assert isinstance(adf_stat, pd.Series)
    assert isinstance(adf_pval, pd.Series)
    assert len(adf_stat) == len(df)
    assert len(adf_pval) == len(df)
    assert adf_stat.name == "adf_stat"
    assert adf_pval.name == "adf_pval"

    # Ensure values are within expected ranges
    valid_stat = adf_stat.dropna()
    valid_pval = adf_pval.dropna()
    assert np.all(np.isfinite(valid_stat))
    assert np.all((valid_pval >= 0) & (valid_pval <= 1))


def test_arch_test_structure_and_validity(ohlcv_sample):
    """
    Test the `arch_test` function for structural correctness and rolling behavior.
    """
    df = ohlcv_sample.copy().head(300)

    arch_stat, arch_pval = arch_test(df, col="close", window_size=60, lags=5)

    # === Structural Checks ===
    assert isinstance(arch_stat, pd.Series)
    assert isinstance(arch_pval, pd.Series)
    assert len(arch_stat) == len(df) - 60
    assert len(arch_pval) == len(df) - 60
    assert arch_stat.index.equals(df.index[60:])
    assert arch_pval.index.equals(df.index[60:])
    assert arch_stat.name == "arch_stat"
    assert arch_pval.name == "arch_pval"

    # === Value Checks ===
    assert np.all(np.isfinite(arch_stat))
    assert np.all((arch_pval >= 0) & (arch_pval <= 1))

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        arch_test(df, col="missing_col", window_size=60)
    with pytest.raises(ValueError):
        arch_test(df, col="close", window_size=0)
    with pytest.raises(ValueError):
        arch_test(df, col="close", window_size=5, lags=10)
    with pytest.raises(ValueError):
        arch_test(df, col="close", window_size=60, lags=0)
    with pytest.raises(ValueError):
        arch_test(df, col="close", window_size=60, ddof=-1)

    # === Side Effect Check ===
    df_copy = df.copy()
    arch_test(df_copy, col="close", window_size=60)
    pd.testing.assert_frame_equal(df, df_copy)



def test_shapiro_wilk_structure_and_validity(ohlcv_sample):
    """
    Test the `shapiro_wilk` function for structure, padding, and valid results.
    """
    df = ohlcv_sample.copy().head(200)
    window_size = 20

    stat, pval = shapiro_wilk(df, col="close", window_size=window_size)

    # === Structural Checks ===
    assert isinstance(stat, pd.Series)
    assert isinstance(pval, pd.Series)
    assert len(stat) == len(df)
    assert len(pval) == len(df)
    assert stat.index.equals(df.index)
    assert pval.index.equals(df.index)
    assert stat.name == f"close_shapiro_stat"
    assert pval.name == f"close_shapiro_pval"

    # === Value Checks ===
    assert stat.iloc[:window_size - 1].isna().all()
    assert pval.iloc[:window_size - 1].isna().all()

    valid_stat = stat.iloc[window_size - 1:].dropna()
    valid_pval = pval.iloc[window_size - 1:].dropna()

    assert np.all((valid_stat > 0) & (valid_stat <= 1))
    assert np.all((valid_pval >= 0) & (valid_pval <= 1))

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        shapiro_wilk(df, col="missing_col", window_size=window_size)

    # === Side Effect Check ===
    df_copy = df.copy()
    shapiro_wilk(df_copy, col="close", window_size=window_size)
    pd.testing.assert_frame_equal(df, df_copy)
