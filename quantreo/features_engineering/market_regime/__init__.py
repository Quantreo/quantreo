import numpy as np
from ..trend import kama


def kama_market_regime(df, col,
                       l1_fast=50, l2_fast=2, l3_fast=30,
                       l1_slow=200, l2_slow=2, l3_slow=30):
    """
    Compute a market regime indicator based on the difference between two KAMA (fast and slow).

    This function calculates two KAMA indicators using two different parameter sets (fast and slow).
    It then derives a market regime signal based on the difference between the fast KAMA and slow KAMA:

    - Returns 1 when the fast KAMA is above the slow KAMA (bullish regime).
    - Returns -1 when the fast KAMA is below the slow KAMA (bearish regime).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the input price series.
    col : str
        Column name on which to apply the KAMA calculation (e.g., 'close').
    l1_fast : int, optional
        Efficiency ratio lookback window for the fast KAMA (default is 50).
    l2_fast : int, optional
        Fastest EMA constant for the fast KAMA (default is 2).
    l3_fast : int, optional
        Slowest EMA constant for the fast KAMA (default is 30).
    l1_slow : int, optional
        Efficiency ratio lookback window for the slow KAMA (default is 200).
    l2_slow : int, optional
        Fastest EMA constant for the slow KAMA (default is 2).
    l3_slow : int, optional
        Slowest EMA constant for the slow KAMA (default is 30).

    Returns
    -------
    pandas.Series
        A Series containing the market regime indicator:
        - 1 for bullish regime
        - -1 for bearish regime
    """
    if col not in df.columns:
        raise ValueError(f"The required column '{col}' is not present in the DataFrame.")

    df_copy = df.copy()

    # Calculate both KAMA values
    df_copy["kama_fast"] = kama(df_copy, col, l1=l1_fast, l2=l2_fast, l3=l3_fast)
    df_copy["kama_slow"] = kama(df_copy, col, l1=l1_slow, l2=l2_slow, l3=l3_slow)

    # Difference & regime detection
    df_copy["kama_diff"] = df_copy["kama_fast"] - df_copy["kama_slow"]
    df_copy["kama_trend"] = np.where(df_copy["kama_diff"] > 0, 1, -1)

    return df_copy["kama_trend"]

