import numpy as np
import pandas as pd
from numba import njit


@njit
def _get_linear_regression_slope(series: np.ndarray) -> float:
    """
    Compute the slope of a linear regression line using a fast implementation with Numba.

    This function calculates the slope of the best-fit line through a 1D array of values
    using the least squares method. It is optimized for performance using the @njit decorator from Numba.

    Parameters
    ----------
    series : np.ndarray
        A one-dimensional NumPy array representing the input time series values.

    Returns
    -------
    float
        The slope of the linear regression line fitted to the input series.

    Notes
    -----
    This function is mainly used internally for rolling or local trend estimation.
    It is not intended to be called directly with a full DataFrame. Use it within a windowed operation.
    """

    n = len(series)
    x = np.arange(n)
    y = series

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x**2

    return numerator / denominator


def linear_slope(df: pd.DataFrame, col: str, window_size: int = 60) -> pd.Series:
    """
    Compute the slope of a linear regression line over a rolling window.

    This function applies a linear regression on a rolling window of a selected column,
    returning the slope of the fitted line at each time step. It uses a fast internal implementation
    (`_get_linear_regression_slope`) for efficient computation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the time series data.
    col : str
        Name of the column on which to compute the slope.
    window_size : int, optional
        Size of the rolling window used to fit the linear regression (default is 60).

    Returns
    -------
    slope_series : pandas.Series
        A Series containing the slope of the regression line at each time step.
        The first (windows_size - 1) values will be NaN due to insufficient data for the initial windows.

    Notes
    -----
    This indicator is useful to assess short- or medium-term price trends.
    A positive slope indicates an upward trend, while a negative slope reflects a downward trend.
    """
    lin_slope = df[col].rolling(window_size).apply(_get_linear_regression_slope, raw=True)
    lin_slope.name = f"linear_slope_{window_size}"
    return lin_slope
