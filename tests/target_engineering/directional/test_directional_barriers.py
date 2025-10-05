import pandas as pd
import numpy as np
import pytest
from quantreo.target_engineering.directional.barriers import double_barrier_labeling, triple_barrier_labeling


def test_double_barrier_labeling(ohlcv_with_time_sample):
    """Test the double_barrier_labeling function."""
    df = ohlcv_with_time_sample.copy().head(300)

    # === Basic functional call ===
    labels = double_barrier_labeling(df)

    # === Structural Checks ===
    assert isinstance(labels, pd.Series)
    assert len(labels) == len(df)
    assert labels.index.equals(df.index)
    assert labels.name == "barrier_label"

    # === Value Checks ===
    unique_vals = set(labels.unique())
    assert unique_vals.issubset({-1, 0, 1}), f"Unexpected labels: {unique_vals}"

    # Should not be all zeros
    assert (labels != 0).any(), "All labels are zero — no TP/SL triggered."

    # === Robustness Checks ===
    # Invalid column name → ValueError
    with pytest.raises(ValueError):
        double_barrier_labeling(df, open_col="not_a_col")

    # Invalid TP/SL values
    with pytest.raises(ValueError):
        double_barrier_labeling(df, tp=-0.01)
    with pytest.raises(ValueError):
        double_barrier_labeling(df, sl=0.01)

    # === Side Effect Check ===
    df_original = df.copy()
    double_barrier_labeling(df_original)
    pd.testing.assert_frame_equal(df, df_original)


def test_triple_barrier_labeling(ohlcv_with_time_sample):
    """Test the triple_barrier_labeling function."""
    df = ohlcv_with_time_sample.copy().head(300)

    # === Basic functional call ===
    labels = triple_barrier_labeling(df, max_duration_h=24)

    # === Structural Checks ===
    assert isinstance(labels, pd.Series)
    assert len(labels) == len(df)
    assert labels.index.equals(df.index)
    assert labels.name == "triple_barrier_label"

    # === Value Checks ===
    unique_vals = set(labels.unique())
    assert unique_vals.issubset({-1, 0, 1}), f"Unexpected labels: {unique_vals}"

    # Must have at least one non-zero label
    assert (labels != 0).any(), "All labels are zero — barriers not triggered or too strict."

    # === Robustness Checks ===
    # Invalid column → error
    with pytest.raises(ValueError):
        triple_barrier_labeling(df, max_duration_h=24, open_col="not_a_col")

    # Invalid tp/sl → errors
    with pytest.raises(ValueError):
        triple_barrier_labeling(df, max_duration_h=24, tp=-0.01)
    with pytest.raises(ValueError):
        triple_barrier_labeling(df, max_duration_h=24, sl=0.01)

    # === Side Effect Check ===
    df_original = df.copy()
    triple_barrier_labeling(df_original, max_duration_h=24)
    pd.testing.assert_frame_equal(df, df_original)
