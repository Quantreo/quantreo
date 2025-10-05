import numpy as np
import pandas as pd
import pytest
from quantreo.target_engineering.magnitude.barriers import continuous_barrier_labeling


def test_continuous_barrier_labeling(ohlcv_with_time_sample):
    """Test the continuous_barrier_labeling function."""
    df = ohlcv_with_time_sample.copy().head(300)

    # === Basic functional call ===
    labels = continuous_barrier_labeling(df, tp=0.015, sl=-0.015, buy=True)

    # === Structural Checks ===
    # Ensure the output is a pandas Series aligned with the input index
    assert isinstance(labels, pd.Series)
    assert len(labels) == len(df)
    assert labels.index.equals(df.index)

    # === Value Checks ===
    # The label should contain only finite values and no NaNs
    assert not labels.isna().any()
    assert np.isfinite(labels).all()

    # Values should be within a reasonable range (±24h max for realistic data)
    assert (labels.abs() < 24 * 60).any() or (labels == 0).all()

    # Last value should be zero by design
    assert labels.iloc[-1] == 0

    # === Logical Checks ===
    # Long positions: positive values = TP hit first, negative = SL hit first
    # Simulate short version and verify sign inversion
    short_labels = continuous_barrier_labeling(df, tp=0.015, sl=-0.015, buy=False)
    assert (labels[labels > 0].shape[0] + labels[labels < 0].shape[0]) > 0
    assert np.all(np.sign(short_labels) == -np.sign(labels))

    # === Robustness Checks ===
    # Missing columns should raise ValueError
    bad_df = df.drop(columns=["high_time"])
    with pytest.raises(ValueError):
        continuous_barrier_labeling(bad_df)

    # Invalid TP or SL values should raise ValueError
    with pytest.raises(ValueError):
        continuous_barrier_labeling(df, tp=-0.01)
    with pytest.raises(ValueError):
        continuous_barrier_labeling(df, sl=0.01)

    # Too short DataFrame should raise ValueError
    with pytest.raises(ValueError):
        continuous_barrier_labeling(df.head(1))

    # === Side Effect Check ===
    # Ensure the original DataFrame remains unchanged after processing
    df_original = df.copy()
    continuous_barrier_labeling(df_original)
    pd.testing.assert_frame_equal(df, df_original)
