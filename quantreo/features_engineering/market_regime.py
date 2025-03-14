import pandas as pd
from trend import kama


def kama_market_regime(df, col, n, m):
    """
    Calculate the market regime using Kaufman's Adaptive Moving Average (KAMA).

    The function computes two KAMA values using two different periods (n and m), then calculates
    the difference between them to determine the market regime. A trend indicator is derived:
    - 1 indicates a positive trend (kama_diff > 0)
    - -1 indicates a negative trend (kama_diff <= 0)

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the price data or numeric series.
    col : str
        Column name on which to apply the KAMA calculation.
    n : int
        Period length for the first KAMA calculation.
    m : int
        Period length for the second KAMA calculation.

    Returns
    -------
    regime_df : pandas.DataFrame
        A copy of the original DataFrame with two additional columns:
            - 'kama_diff': the difference between KAMA computed with period m and period n.
            - 'kama_trend': an indicator of market trend (1 for positive, -1 for negative).
    """
    # Check that the required column exists in the DataFrame
    if col not in df.columns:
        raise ValueError(f"The required column '{col}' is not present in the DataFrame.")

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Calculate KAMA for both periods.
    df_copy = kama(df_copy, col, n)
    df_copy = kama(df_copy, col, m)

    # Calculate the difference between the two KAMA values.
    df_copy["kama_diff"] = df_copy[f"kama_{m}"] - df_copy[f"kama_{n}"]

    # Determine the market regime based on the difference:
    # 1 if kama_diff > 0, -1 otherwise.
    df_copy["kama_trend"] = -1
    df_copy.loc[df_copy["kama_diff"] > 0, "kama_trend"] = 1

    return df_copy
