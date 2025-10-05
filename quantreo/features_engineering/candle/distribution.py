import pandas as pd
import numpy as np
from numba import njit


@njit
def _close_percentage_in_range(close_window: np.ndarray, start_pct: float, end_pct: float) -> float:
    """
    Compute the percentage of values within a sub-range of the window, based on relative position
    between the local min and max (low and high) of the window.

    Parameters
    ----------
    close_window : np.ndarray
        One-dimensional array of close prices (rolling window).
    start_pct : float
        Start of the range as a percentage of (high - low). Example: 0.25 = 25%.
    end_pct : float
        End of the range as a percentage of (high - low). Example: 0.75 = 75%.

    Returns
    -------
    float
        Percentage of values within the specified sub-range of the price interval.
        Returns 0.0 if no valid (non-NaN) values are found in the window.
    """
    low = np.min(close_window)
    high = np.max(close_window)
    start_threshold = low + start_pct * (high - low)
    end_threshold = low + end_pct * (high - low)

    count = 0
    total = 0

    for price in close_window:
        if not np.isnan(price):
            total += 1
            if start_threshold <= price <= end_threshold:
                count += 1

    return (count / total) * 100 if total > 0 else 0.0


def price_distribution(
    df: pd.DataFrame,
    col: str,
    window_size: int = 60,
    start_percentage: float = 0.25,
    end_percentage: float = 0.75,
) -> pd.Series:
    """
    Compute the percentage of close prices within a relative range of their local low-high interval,
    over a rolling window.

    This function calculates, for each window, how many values lie within a given percentage band
    of the [low, high] range. It is useful to detect price compression or expansion around a zone.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series data.
    col : str
        Name of the column containing the close prices.
    window_size : int, optional
        Size of the rolling window (default is 60).
    start_percentage : float, optional
        Start of the relative range as a percentage of (high - low). Default is 0.25 (25%).
    end_percentage : float, optional
        End of the relative range as a percentage of (high - low). Default is 0.75 (75%).

    Returns
    -------
    pd.Series
        Series with the same index as the input, containing the computed percentage values for each window.
        First (window_size - 1) rows will be NaN.
    """
    return (
        df[col]
        .rolling(window_size)
        .apply(lambda x: _close_percentage_in_range(x, start_percentage, end_percentage), raw=True)
    )
