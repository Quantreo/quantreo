import pandas as pd
import numpy as np


def compute_spread(df, high_col='high', low_col='low'):
    """
    Compute the spread between the high and low price columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing price data.
    high_col : str, optional
        Column name for the high prices (default is 'high').
    low_col : str, optional
        Column name for the low prices (default is 'low').

    Returns
    -------
    spread_series : pandas.Series
        A Series indexed the same as `df`, containing the spread values.
    """
    # Check that the necessary columns exist in the DataFrame
    for col in [high_col, low_col]:
        if col not in df.columns:
            raise ValueError(f"The required column '{col}' is not present in the DataFrame.")

    # Compute the spread
    spread_series = df[high_col] - df[low_col]

    # Return as a Series with a clear name
    return pd.Series(spread_series, name="spread", index=df.index)
