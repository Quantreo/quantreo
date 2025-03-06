import numpy as np
import pandas as pd
from numba import njit


@njit(nogil=True)
def yang_zhang_estimator(high, low, open_, close, window_size):
    n = high.shape[0]
    vol = np.empty(n)

    # Fill the first values (for which there isn't enough data) with NaN.
    for i in range(window_size):
        vol[i] = np.nan

    # Compute rolling volatility over the sliding window
    for i in range(window_size, n):
        sum_val = 0.0
        N = window_size + 1  # Number of elements in the window: from i-window_size to i inclusive

        for j in range(i - window_size, i + 1):
            term1 = np.log(high[j] / close[j]) * np.log(high[j] / open_[j])
            term2 = np.log(low[j] / close[j]) * np.log(low[j] / open_[j])
            sum_val += term1 + term2
        vol[i] = np.sqrt(sum_val / N)
    return vol


def moving_yang_zhang_estimator(df, window_size=30, high_col='high', low_col='low', open_col='open', close_col='close'):
    """
    Calculate Yang-Zhang volatility estimator using numpy operations with Numba acceleration.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the price data.
    window_size : int, optional
        The number of periods to include in the rolling calculation (default is 30).
    high_col : str, optional
        Column name for the high prices (default is 'high').
    low_col : str, optional
        Column name for the low prices (default is 'low').
    open_col : str, optional
        Column name for the open prices (default is 'open').
    close_col : str, optional
        Column name for the close prices (default is 'close').

    Returns
    -------
    volatility_series : pandas.Series
        A Series indexed the same as `df`, containing the rolling Yang-Zhang volatility.
        The first `window_size` rows will be NaN because there is insufficient data
        to compute the volatility in those windows.
    """
    # Check that the necessary columns exist in the DataFrame
    for col in [high_col, low_col, open_col, close_col]:
        if col not in df.columns:
            raise ValueError(f"The required column '{col}' is not present in the DataFrame.")

    # Convert the specified columns to NumPy arrays
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    open_ = df[open_col].to_numpy()
    close = df[close_col].to_numpy()

    # Calculate the volatility using the Numba-accelerated function
    vol_array = yang_zhang_estimator(high, low, open_, close, window_size)

    # Create a Series and add the calculated volatility as a new column
    series = pd.Series(vol_array, name="rolling_volatility_yang_zhang", index=df.index)

    return series
