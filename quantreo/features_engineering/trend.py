import numpy as np
import pandas as pd
from numba import njit


def sma(df: pd.DataFrame, col: str, window: int = 30) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA) using Pandas rolling.mean,
    following the standard conventions of our library.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the input data.
    col : str
        Name of the column on which to compute the SMA.
    window : int, optional
        The window size for computing the SMA (default is 30).

    Returns
    -------
    sma_series : pandas.Series
        A Series indexed the same as the input DataFrame, containing the SMA values.
        The first (window - 1) entries will be NaN due to insufficient data.
    """
    # Verify that the required column exists
    if col not in df.columns:
        raise ValueError(f"The column '{col}' is not present in the DataFrame.")

    # Compute the SMA using Pandas' rolling.mean()
    sma_series = df[col].rolling(window=window).mean()

    # Rename the series for clarity
    sma_series.name = f"sma_{window}"

    return sma_series


def kama(df: pd.DataFrame, col: str, l1: int = 10, l2: int = 2, l3: int = 30) -> pd.Series:
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA) for a specified column in a DataFrame.

    KAMA adapts to market noise by adjusting its smoothing constant based on an efficiency ratio.
    The efficiency ratio is computed over a rolling window of length `l1` as:
        ER = |close - close.shift(l1)| / (rolling sum of |close - close.shift(1)| over l1 periods)
    The smoothing constant is then calculated as:
        sc = [ ER * (2/(l2+1) - 2/(l3+1)) + 2/(l3+1) ]^2
    and KAMA is computed recursively:
        KAMA(i) = KAMA(i-1) + sc(i) * (close(i) - KAMA(i-1))

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the price data.
    col : str
        Column name on which to compute the KAMA.
    l1 : int, optional
        Rolling window length for computing the efficiency ratio (default is 10).
    l2 : int, optional
        Parameter for the fastest EMA constant (default is 2).
    l3 : int, optional
        Parameter for the slowest EMA constant (default is 30).

    Returns
    -------
    pandas.Series
        A Series containing the computed KAMA values, indexed the same as `df` and named "kama".
        The first (l1 - 1) values will likely be NaN due to insufficient data.
    """
    # Verify that the specified column exists
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Convert the column to float for consistency
    close_series = df[col].astype(float)
    close_values = close_series.values
    n = len(close_values)

    # Calculate volatility.md: absolute difference between consecutive close values
    vol = pd.Series(np.abs(close_series - close_series.shift(1)), index=close_series.index)

    # Efficiency ratio numerator: absolute difference between current close and close l1 periods ago
    er_num = np.abs(close_series - close_series.shift(l1))
    # Efficiency ratio denominator: rolling sum of volatility.md over a window of l1 periods
    er_den = vol.rolling(window=l1, min_periods=l1).sum()

    # Compute efficiency ratio; fill NaN (or division by zero) with 0
    efficiency_ratio = (er_num / er_den).fillna(0)

    # Compute the smoothing constant, converting the result to a NumPy array for fast access
    sc = ((efficiency_ratio * (2.0 / (l2 + 1) - 2.0 / (l3 + 1)) + 2.0 / (l3 + 1)) ** 2).values

    # Initialize an array to hold the KAMA values
    kama_values = np.full(n, np.nan)
    first_value = True

    # Recursive calculation of KAMA
    for i in range(n):
        if np.isnan(sc[i]):
            kama_values[i] = np.nan
        elif first_value:
            # Set the initial KAMA value as the first close available
            kama_values[i] = close_values[i]
            first_value = False
        else:
            kama_values[i] = kama_values[i - 1] + sc[i] * (close_values[i] - kama_values[i - 1])

    return pd.Series(kama_values, index=df.index, name="kama")


