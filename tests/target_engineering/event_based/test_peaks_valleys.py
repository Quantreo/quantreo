import numpy as np
import pandas as pd
import pytest
from quantreo.target_engineering.event_based.peaks_valleys import detect_peaks_valleys


def test_detect_peaks_valleys(ohlcv_sample):
    """Test the detect_peaks_valleys function."""
    df = ohlcv_sample.copy().head(300)
    df["close"] = np.sin(np.linspace(0, 12 * np.pi, len(df)))  # sinus → parfait pour tests

    # === Basic functional call ===
    result = detect_peaks_valleys(df, col="close", distance=5)

    # === Structural Checks ===
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == "label"

    # === Value Checks ===
    # Vérifie que les labels sont bien dans {-1, 0, 1}
    unique_labels = set(result.dropna().unique())
    assert unique_labels.issubset({-1, 0, 1})

    # Vérifie qu'il y a au moins un pic et une vallée
    assert (result == 1).sum() > 0
    assert (result == -1).sum() > 0

    # === Logical Checks ===
    # Les pics doivent correspondre à des maximums locaux
    peaks_idx = result[result == 1].index
    for idx in peaks_idx:
        if idx not in [df.index[0], df.index[-1]]:
            assert df.loc[idx, "close"] >= df["close"].shift(1).loc[idx]
            assert df.loc[idx, "close"] >= df["close"].shift(-1).loc[idx]

    # Les vallées doivent correspondre à des minimums locaux
    valleys_idx = result[result == -1].index
    for idx in valleys_idx:
        if idx not in [df.index[0], df.index[-1]]:
            assert df.loc[idx, "close"] <= df["close"].shift(1).loc[idx]
            assert df.loc[idx, "close"] <= df["close"].shift(-1).loc[idx]

    # === Robustness Checks ===
    # Mauvais nom de colonne → doit lever une erreur
    with pytest.raises(ValueError):
        detect_peaks_valleys(df, col="not_a_col")

    # Le DataFrame d’origine ne doit pas être modifié
    df_original = df.copy()
    detect_peaks_valleys(df_original, col="close", distance=5)
    pd.testing.assert_frame_equal(df, df_original)
