import pandas as pd
import numpy as np
from quantreo.data_aggregation.bar_building.tick_bars import ticks_to_tick_bars


def test_ticks_to_tick_bars(ticks_sample):
    """Test the ticks_to_tick_bars function."""
    df = ticks_sample.copy()

    # === Basic functional call ===
    bars = ticks_to_tick_bars(df, tick_per_bar=1000)

    # === Structural Checks ===
    # Ensure the function returns a valid DataFrame with expected columns
    assert isinstance(bars, pd.DataFrame)
    assert len(bars) > 0
    expected_cols = [
        "open", "high", "low", "close", "volume",
        "number_ticks", "duration_minutes", "high_time", "low_time"
    ]
    assert all(col in bars.columns for col in expected_cols)
    assert bars.index.name == "time"
    assert pd.api.types.is_datetime64_any_dtype(bars.index)

    # === Value Checks ===
    # No missing or infinite values in key columns
    for col in ["open", "high", "low", "close", "volume", "number_ticks"]:
        assert not bars[col].isna().any()
        assert np.isfinite(bars[col]).all()

    # The total tick volume should approximately equal the sum of bar volumes
    vol_ticks = df["volume"].sum()
    vol_bars = bars["volume"].sum()
    assert np.isclose(vol_ticks, vol_bars, rtol=1e-3)

    # The total number of ticks should match the sum of number_ticks (allowing for rounding)
    total_ticks = len(df)
    counted_ticks = bars["number_ticks"].sum()
    assert abs(total_ticks - counted_ticks) <= bars["number_ticks"].max()

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
    ticks_to_tick_bars(df_original, tick_per_bar=1000)
    pd.testing.assert_frame_equal(df, df_original)
