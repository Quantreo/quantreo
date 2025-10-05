import pytest
import numpy as np
import pandas as pd

from quantreo.features_engineering.trend.slopes import linear_slope


def test_linear_slope(ohlcv_sample):
    """Test the linear_slope function."""
    df = ohlcv_sample.copy().head(300)

    result = linear_slope(df=df, col="close", window_size=60)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "linear_slope_60"

    # === Value Checks ===
    assert result.iloc[:59].isna().all()  # premières valeurs NaN
    valid = result.iloc[59:]
    assert not valid.isna().any()
    assert not np.isinf(valid).any()

    # Vérifie que les pentes sont proches de zéro sur des données sans tendance
    flat_df = pd.DataFrame({"close": np.ones(200)})
    flat_slope = linear_slope(flat_df, col="close", window_size=60)
    assert np.allclose(flat_slope.dropna(), 0, atol=1e-10)

    # Vérifie qu'une série croissante donne des pentes positives
    trend_df = pd.DataFrame({"close": np.arange(200)})
    trend_slope = linear_slope(trend_df, col="close", window_size=60)
    assert (trend_slope.dropna() > 0).all()

    # Vérifie qu'une série décroissante donne des pentes négatives
    trend_df_down = pd.DataFrame({"close": -np.arange(200)})
    trend_slope_down = linear_slope(trend_df_down, col="close", window_size=60)
    assert (trend_slope_down.dropna() < 0).all()

    # === Robustness Checks ===
    with pytest.raises(ValueError):
        linear_slope(df, col="not_a_col")

    # === Side Effect Check ===
    df_original = df.copy()
    linear_slope(df_original, col="close")
    pd.testing.assert_frame_equal(df, df_original)