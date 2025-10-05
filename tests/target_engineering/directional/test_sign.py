import pytest
import numpy as np
import pandas as pd
from quantreo.target_engineering.directional.sign import future_returns_sign


def test_future_returns_sign(ohlcv_sample):
    """Test the future_returns_sign function."""
    df = ohlcv_sample.copy().head(300)

    # === Basic functional call ===
    labels = future_returns_sign(df, close_col="close", window_size=10, log_return=True)

    # === Structural Checks ===
    # Ensure it returns a pandas Series of same length and index
    assert isinstance(labels, pd.Series)
    assert len(labels) == len(df)
    assert labels.index.equals(df.index)

    # === Value Checks ===
    # Must contain only binary (0/1) or unique custom labels
    unique_vals = set(labels.unique())
    assert unique_vals.issubset({0, 1})

    # Ensure NaN alignment matches the future_returns tail NaN window
    tail = labels.iloc[-10:]
    assert (tail == 0).all() or tail.isna().all()

    # Ensure distribution is not degenerate
    assert labels.sum() > 0, "No positive labels detected — check return computation"
    assert (labels == 0).sum() > 0, "No zero labels detected — check return computation"

    # === Coherence Checks ===
    # Positive labels should correspond to price increases
    price_diff = df["close"].shift(-10) - df["close"]
    same_direction = ((price_diff > 0) & (labels == 1)) | ((price_diff <= 0) & (labels == 0))
    assert same_direction.mean() > 0.8  # at least 80% consistency

    # === Custom Labels ===
    custom_labels = future_returns_sign(df, close_col="close", window_size=10, log_return=False,
                                        positive_label="UP", negative_label="DOWN")
    assert set(custom_labels.unique()).issubset({"UP", "DOWN"})
    assert len(custom_labels) == len(df)

    # === Robustness Checks ===
    # Missing column
    with pytest.raises(ValueError):
        future_returns_sign(df, close_col="not_a_col")

    # Zero or negative window_size
    with pytest.raises(Exception):
        future_returns_sign(df, window_size=0)

    # === Side Effect Check ===
    df_original = df.copy()
    future_returns_sign(df_original, close_col="close", window_size=10)
    pd.testing.assert_frame_equal(df, df_original)
