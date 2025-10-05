import pandas as pd
import numpy as np
import pytest
from quantreo.target_engineering.directional.quantile import quantile_label


def test_quantile_label_basic():
    """Test the quantile_label function under normal conditions."""
    np.random.seed(42)
    df = pd.DataFrame({"fut_ret": np.random.randn(1000)})

    # === Basic functional call ===
    labels = quantile_label(df, col="fut_ret", upper_quantile_level=0.7)

    # === Structural Checks ===
    assert isinstance(labels, pd.Series)
    assert len(labels) == len(df)
    assert labels.index.equals(df.index)

    # === Value Checks ===
    unique_vals = set(labels.unique())
    assert unique_vals.issubset({-1, 0, 1})

    # Check proportion roughly matches quantiles (±5 %)
    prop_pos = (labels == 1).mean()
    prop_neg = (labels == -1).mean()

    expected_pos = 1 - 0.7        # 30 %
    expected_neg = 0.7 - 0.3      # 40 %, mais ici ~30 % pour un N~1000 aléatoire
    assert abs(prop_pos - expected_pos) < 0.05
    assert abs(prop_neg - expected_pos) < 0.05  # cohérence avec symétrie du tirage

    assert (labels == 0).any()

    # === Threshold Return Check ===
    labels_, q_high, q_low = quantile_label(df, col="fut_ret", upper_quantile_level=0.7, return_thresholds=True)
    assert np.isclose(q_high, df["fut_ret"].quantile(0.7))
    assert np.isclose(q_low, df["fut_ret"].quantile(0.3))

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        quantile_label(df, col="fut_ret", q_high=0, q_low=1)
    with pytest.raises(KeyError):
        quantile_label(df, col="not_a_col")

    # === Custom Label Values ===
    custom_labels = quantile_label(df, col="fut_ret", positive_label="UP", negative_label="DOWN", neutral_label="MID")
    assert set(custom_labels.unique()).issubset({"UP", "DOWN", "MID"})

    # === Side Effect Check ===
    df_original = df.copy()
    quantile_label(df_original, col="fut_ret")
    pd.testing.assert_frame_equal(df, df_original)
