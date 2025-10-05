import pytest
import numpy as np
import pandas as pd

from quantreo.features_engineering.volatility.close_to_close import close_to_close_volatility


def test_close_to_close_volatility(ohlcv_sample):
    """Test the close_to_close_volatility function."""
    df = ohlcv_sample.copy().head(300)

    result = close_to_close_volatility(df=df, close_col="close", window_size=30)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "close_to_close_vol"

    # === Value Checks ===
    # Les premières valeurs doivent être NaN (fenêtre incomplète)
    assert result.iloc[:30].isna().all()
    valid = result.iloc[30:]
    assert not valid.isna().any()
    assert not np.isinf(valid).any()
    # Volatilité toujours >= 0
    assert (valid >= 0).all()

    # Vérifie cohérence avec un calcul manuel
    manual_log_returns = df["close"].pct_change().apply(lambda x: np.log(1 + x))
    manual_vol = manual_log_returns.rolling(30).std()
    pd.testing.assert_series_equal(result, manual_vol, check_names=False)

    # === Robustness Checks ===
    # Colonne manquante
    with pytest.raises(ValueError):
        close_to_close_volatility(df, close_col="not_a_col")

    # Test sur fenêtre invalide (trop petite)
    result_short = close_to_close_volatility(df, close_col="close", window_size=1)
    assert isinstance(result_short, pd.Series)
    assert len(result_short) == len(df)

    # === Behavior Checks ===
    # Série constante => volatilité nulle
    flat_df = pd.DataFrame({"close": np.ones(100)})
    flat_vol = close_to_close_volatility(flat_df, close_col="close", window_size=30)
    assert np.allclose(flat_vol.dropna(), 0.0, atol=1e-12)

    # === Side Effect Check ===
    df_original = df.copy()
    close_to_close_volatility(df_original, close_col="close")
    pd.testing.assert_frame_equal(df, df_original)
