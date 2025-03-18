import numpy as np
import math
import pandas as pd
from numba import njit


def close_to_close_volatility(df: pd.DataFrame, close_col: str = 'close', window_size: int = 30) -> pd.Series:
    """
    Calculate the rolling close-to-close volatility.md using standard deviation.

    This method computes the rolling standard deviation of the log returns,
    which represents the close-to-close volatility.md over a specified window.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the price data.
    window_size : int, optional
        The number of periods to include in the rolling calculation (default is 30).
    close_col : str, optional
        Column name for the closing prices (default is 'close').

    Returns
    -------
    volatility_series : pd.Series
        A Series indexed the same as `df`, containing the rolling close-to-close volatility.md.
        The first `window_size` rows will be NaN because there is insufficient data
        to compute the volatility.md in those windows.
    """
    # Ensure the required column exists
    if close_col not in df.columns:
        raise ValueError(f"The required column '{close_col}' is not present in the DataFrame.")

    # Compute log returns
    log_returns = df[close_col].pct_change().apply(lambda x: np.log(1 + x))

    # Compute rolling standard deviation of log returns
    volatility_series = log_returns.rolling(window=window_size, min_periods=window_size).std()

    # Name the resulting column
    volatility_series.name = "close_to_close_vol"

    return volatility_series


@njit(nogil=True)
def _rogers_satchell_estimator(high, low, open_, close, window_size):
    n = high.shape[0]
    vol = np.empty(n)

    # Fill the first values (for which there isn't enough data) with NaN.
    for i in range(window_size):
        vol[i] = np.nan

    # Compute rolling volatility.md over the sliding window
    for i in range(window_size, n):
        sum_val = 0.0
        N = window_size + 0

        for j in range(i - window_size, i):
            term1 = np.log(high[j] / close[j]) * np.log(high[j] / open_[j])
            term2 = np.log(low[j] / close[j]) * np.log(low[j] / open_[j])
            sum_val += term1 + term2
        vol[i] = np.sqrt(sum_val / N)
    return vol


def rogers_satchell_volatility(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low',
                          open_col: str = 'open', close_col: str = 'close', window_size: int = 30) -> pd.Series:
    """
    Calculate Rogers-Satchell volatility.md estimator using numpy operations with Numba acceleration.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the price data.
    high_col : str, optional
        Column name for the high prices (default is 'high').
    low_col : str, optional
        Column name for the low prices (default is 'low').
    open_col : str, optional
        Column name for the open prices (default is 'open').
    close_col : str, optional
        Column name for the close prices (default is 'close').
    window_size : int, optional
        The number of periods to include in the rolling calculation (default is 30).

    Returns
    -------
    volatility_series : pandas.Series
        A Series indexed the same as `df`, containing the rolling Rogers-Satchell volatility.md.
        The first `window_size` rows will be NaN because there is insufficient data
        to compute the volatility.md in those windows.
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

    # Calculate the volatility.md using the Numba-accelerated function
    vol_array = _rogers_satchell_estimator(high, low, open_, close, window_size)

    # Create a Series and add the calculated volatility.md as a new column
    series = pd.Series(vol_array, name="rogers_satchell_vol", index=df.index)

    return series


@njit(nogil=True)
def _parkinson_estimator(high, low, window_size):
    """
    Compute Parkinson's volatility.md estimator over a rolling window.

    Parameters
    ----------
    high : np.ndarray
        Array of high prices.
    low : np.ndarray
        Array of low prices.
    window_size : int
        Rolling window size.

    Returns
    -------
    vol : np.ndarray
        Array containing the rolling Parkinson volatility.md.
    """
    n = high.shape[0]
    vol = np.empty(n)

    # Fill the first values (for which there isn't enough data) with NaN.
    for i in range(window_size):
        vol[i] = np.nan

    # Compute rolling volatility.md over the sliding window
    for i in range(window_size, n):
        sum_squared = 0.0
        N = window_size  # Number of elements in the window

        for j in range(i - window_size, i):
            sum_squared += np.log(high[j] / low[j]) ** 2

        vol[i] = math.sqrt((1 / (4 * N * math.log(2))) * sum_squared)

    return vol


def parkinson_volatility(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low', window_size: int = 30)\
                        -> pd.Series:
    """
    Calculate Parkinson's volatility.md estimator using numpy operations with Numba acceleration.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the price data.
    high_col : str, optional
        Column name for the high prices (default is 'high').
    low_col : str, optional
        Column name for the low prices (default is 'low').
    window_size : int, optional
        The number of periods to include in the rolling calculation (default is 30).

    Returns
    -------
    volatility_series : pandas.Series
        A Series indexed the same as `df`, containing the rolling Parkinson volatility.md.
        The first `window_size` rows will be NaN because there is insufficient data
        to compute the volatility.md in those windows.
    """
    # Check that the necessary columns exist in the DataFrame
    for col in [high_col, low_col]:
        if col not in df.columns:
            raise ValueError(f"The required column '{col}' is not present in the DataFrame.")

    # Convert the specified columns to NumPy arrays
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()

    # Calculate the volatility.md using the Numba-accelerated function
    vol_array = _parkinson_estimator(high, low, window_size)

    # Create a Series and add the calculated volatility.md as a new column
    series = pd.Series(vol_array, name="rolling_volatility_vol", index=df.index)

    return series


@njit(nogil=True)
def _yang_zhang_estimator(high, low, open_, close, window_size, k=0.34):
    """
    Compute the Yang-Zhang volatility.md estimator using a single-pass approach,
    similar to the Rogers-Satchell method.

    Parameters
    ----------
    high : np.array
        High prices.
    low : np.array
        Low prices.
    open_ : np.array
        Open prices.
    close : np.array
        Close prices.
    window_size : int
        The rolling window size for the computation.
    k : float, optional
        The weighting factor for the open-to-close variance, derived from
        empirical research by Yang & Zhang (2000). The default value is 0.34.

    Returns
    -------
    np.array
        An array containing the estimated Yang-Zhang volatility.md over the given window.
        The first `window_size` elements are NaN due to insufficient data.
    """
    n = high.shape[0]
    vol = np.empty(n)
    vol[:window_size] = np.nan  # Fill first `window_size` elements with NaN

    # Iterating over each time step
    for i in range(window_size, n):
        sum_oc = 0.0   # Open-to-close variance (σ_O²)
        sum_cc = 0.0   # Close-to-close variance (σ_C²)
        sum_rs = 0.0   # Rogers-Satchell variance (σ_RS²)
        N = window_size  # Number of periods in the rolling window

        # Compute variance components over the rolling window
        for j in range(i - window_size, i):
            # Rogers-Satchell term
            term1 = np.log(high[j] / close[j]) * np.log(high[j] / open_[j])
            term2 = np.log(low[j] / close[j]) * np.log(low[j] / open_[j])
            sum_rs += (term1 + term2)

            # Open-to-close variance (σ_O²)
            diff_oc = np.log(open_[j] / close[j])  # log(Open_j / Close_j)
            sum_oc += diff_oc * diff_oc

            # Close-to-close variance (σ_C²)
            diff_cc = np.log(close[j] / open_[j])  # log(Close_j / Open_j)
            sum_cc += diff_cc * diff_cc

        # Compute variances
        sigma_oc = sum_oc / N
        sigma_cc = sum_cc / N
        sigma_rs = sum_rs / N

        # Yang-Zhang volatility.md formula
        yz_vol = np.sqrt(sigma_oc + k * sigma_cc + (1 - k) * sigma_rs)
        vol[i] = yz_vol

    return vol


def yang_zhang_volatility(df: pd.DataFrame, window_size: int = 30, high_col: str = 'high', low_col: str = 'low',
                          open_col: str = 'open', close_col: str = 'close', k: float = 0.34) -> pd.Series:
    """
    Compute the Yang-Zhang volatility.md estimator using a rolling window.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing OHLC price data.
    window_size : int, optional
        The number of periods used in the rolling calculation (default = 30).
    high_col : str, optional
        Column name for high prices (default = 'high').
    low_col : str, optional
        Column name for low prices (default = 'low').
    open_col : str, optional
        Column name for open prices (default = 'open').
    close_col : str, optional
        Column name for close prices (default = 'close').
    k : float, optional
        The weighting parameter for the open-to-close variance component, as described
        in Yang & Zhang (2000). Empirical research suggests 0.34 as the optimal value.

    Returns
    -------
    pd.Series
        A Series containing the rolling Yang-Zhang volatility.md, indexed like `df`.
        The first `window_size` rows are NaN due to insufficient data.
    """
    # Validate required columns
    for col in [high_col, low_col, open_col, close_col]:
        if col not in df.columns:
            raise ValueError(f"The required column '{col}' is not present in the DataFrame.")

    # Convert to NumPy arrays for efficient computation
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    open_ = df[open_col].to_numpy()
    close = df[close_col].to_numpy()

    # Compute volatility.md using Numba-accelerated function
    vol_array = _yang_zhang_estimator(high, low, open_, close, window_size, k)

    # Convert to pandas Series and return
    return pd.Series(vol_array, name="yang_zhang_vol", index=df.index)
