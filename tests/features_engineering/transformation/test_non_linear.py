import pytest
import numpy as np
import pandas as pd
from quantreo.features_engineering.transformation.non_linear import fisher_transform, logit_transform, neg_log_transform


def test_fisher_transform(ohlcv_sample):
    """
    Test the `fisher_transform` function.
    """
    df = ohlcv_sample.copy().head(200)
    result = fisher_transform(df, high_col="high", low_col="low", window_size=10)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "fisher"

    # === Value Checks ===
    assert result.iloc[:9].isna().all()
    valid = result.iloc[9:].dropna()
    assert np.all(np.isfinite(valid))
    # Should be roughly centered around 0 for normalized prices
    assert abs(valid.mean()) < 3

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        fisher_transform(df, high_col="missing_high", low_col="low")
    with pytest.raises(ValueError):
        fisher_transform(df.assign(high="non_numeric"), high_col="high", low_col="low")

    # === Side Effect Check ===
    df_copy = df.copy()
    fisher_transform(df_copy, high_col="high", low_col="low")
    pd.testing.assert_frame_equal(df, df_copy)


def test_logit_transform(ohlcv_sample):
    """
    Test the `logit_transform` function.
    """
    df = ohlcv_sample.copy().head(100)
    df["ratio"] = np.linspace(0.01, 0.99, 100)
    result = logit_transform(df, col="ratio")

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "ratio_logit"

    # === Value Checks ===
    assert np.all(np.isfinite(result))
    # The logit of 0.5 should be near 0
    mid_value = result.iloc[49]
    assert abs(mid_value) < 0.05

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        logit_transform(df, col="missing_col")
    with pytest.raises(ValueError):
        logit_transform(df.assign(ratio="text"), col="ratio")

    # === Side Effect Check ===
    df_copy = df.copy()
    logit_transform(df_copy, col="ratio")
    pd.testing.assert_frame_equal(df, df_copy)


def test_neg_log_transform(ohlcv_sample):
    """
    Test the `neg_log_transform` function.
    """
    df = ohlcv_sample.copy().head(100)
    df["pval"] = np.linspace(0.001, 1, 100)
    result = neg_log_transform(df, col="pval")

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "pval_neg_log"

    # === Value Checks ===
    assert np.all(np.isfinite(result))
    # As pval approaches 0, -log(p) increases
    assert result.iloc[0] > result.iloc[-1]
    assert result.max() > 0

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        neg_log_transform(df, col="missing_col")
    with pytest.raises(ValueError):
        neg_log_transform(df.assign(pval="non_numeric"), col="pval")

    # === Side Effect Check ===
    df_copy = df.copy()
    neg_log_transform(df_copy, col="pval")
    pd.testing.assert_frame_equal(df, df_copy)