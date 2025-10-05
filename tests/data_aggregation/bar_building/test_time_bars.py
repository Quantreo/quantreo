import numpy as np
import pandas as pd
from quantreo.data_aggregation.bar_building.time_bars import ticks_to_time_bars


def test_ticks_to_time_bars(ticks_sample):
    """Test the ticks_to_time_bars function."""
    df = ticks_sample.copy()

    # === Basic functional call ===
    bars = ticks_to_time_bars(df, resample_factor="30min")

    # === Structural Checks ===
    # Ensure the function returns a valid DataFrame with expected columns
    assert isinstance(bars, pd.DataFrame)
    assert len(bars) > 0
    expected_cols = [
        "open", "high", "low", "close", "volume",
        "number_ticks", "high_time", "low_time"
    ]
    assert all(col in bars.columns for col in expected_cols)
    assert bars.index.name == "time"
    assert pd.api.types.is_datetime64_any_dtype(bars.index)

    # === Value Checks ===
    # No missing or infinite values in key columns
    for col in ["open", "high", "low", "close", "volume", "number_ticks"]:
        assert not bars[col].isna().any()
        assert np.isfinite(bars[col]).all()

    # Volume must be positive
    assert (bars["volume"] > 0).all()

    # Number of ticks per bar must be positive
    assert (bars["number_ticks"] > 0).all()

    # === Logical Checks ===
    # OHLC hierarchy: high ≥ open/close and low ≤ open/close
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()

    # Timestamps of high_time and low_time should be valid and within logical range
    assert pd.api.types.is_datetime64_any_dtype(bars["high_time"])
    assert pd.api.types.is_datetime64_any_dtype(bars["low_time"])
    assert (bars["high_time"] >= bars.index[0]).all()
    assert (bars["low_time"] >= bars.index[0]).all()

    # === Side Effect Check ===
    # Ensure that the original DataFrame remains unchanged after processing
    df_original = df.copy()
    ticks_to_time_bars(df_original, resample_factor="30min")
    pd.testing.assert_frame_equal(df, df_original)
