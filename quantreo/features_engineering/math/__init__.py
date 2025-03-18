import pandas as pd
import numpy as np
from numba import njit
from typing import Tuple
import math
import warnings


def derivatives(df: pd.DataFrame, col: str) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate the first (velocity) and second (acceleration) derivatives of a specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    col : str
        The name of the column for which the derivatives are computed.

    Returns
    -------
    velocity_series : pandas.Series
        The first derivative (velocity) of the specified column.
    acceleration_series : pandas.Series
        The second derivative (acceleration) of the specified column.
    """
    # Verify that the column exists in the DataFrame
    if col not in df.columns:
        raise ValueError(f"The column '{col}' is not present in the DataFrame.")

    # Compute the first derivative (velocity) and fill missing values with 0
    velocity_series = df[col].diff().fillna(0)
    # Compute the second derivative (acceleration) based on velocity
    acceleration_series = velocity_series.diff().fillna(0)

    return velocity_series, acceleration_series


def log_pct(df: pd.DataFrame, col: str, window_size: int) -> pd.Series:
    """
    Apply a logarithmic transformation to a specified column in a DataFrame and calculate
    the percentage change of the log-transformed values over a given window size.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the column to be logarithmically transformed.
    col : str
        The name of the column to which the logarithmic transformation is applied.
    window_size : int
        The window size over which to calculate the percentage change of the log-transformed values.

    Returns
    -------
    pd.Series
        A Series containing the rolling log returns over `n` periods.
    """
    df_copy = df.copy()
    df_copy[f"log_{col}"] = np.log(df_copy[col])
    df_copy[f"ret_log_{window_size}"] = df_copy[f"log_{col}"].pct_change(window_size)

    return df_copy[f"ret_log_{window_size}"]


def auto_corr(df: pd.DataFrame, col: str, window_size: int = 50, lag: int = 10) -> pd.Series:
    """
    Calculate the rolling autocorrelation for a specified column in a DataFrame.

    This function computes the autocorrelation of the values in `col` over a rolling window of size `n`
    with a specified lag. The autocorrelation is calculated using Pandas' rolling.apply() method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data.
    col : str
        The name of the column for which to calculate autocorrelation.
    window_size : int, optional
        The window size for the rolling calculation (default is 50).
    lag : int, optional
        The lag value used when computing autocorrelation (default is 10).

    Returns
    -------
    pd.Series
        A Series containing the rolling autocorrelation values.
    """
    df_copy = df.copy()
    col_name = f'autocorr_{lag}'
    df_copy[col_name] = df_copy[col].rolling(window=window_size, min_periods=window_size).apply(
        lambda x: x.autocorr(lag=lag), raw=False)
    return df_copy[col_name]


@njit
def _std_numba(x):
    """
    Fonction from the hurst library (https://github.com/Mottl/hurst/)
    of Dmitry A. Mottl (pimped using numba).

    Copyright (c) 2017 Dmitry A. Mottl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    n = len(x)
    if n <= 1:
        return np.nan
    mean = 0.0
    for i in range(n):
        mean += x[i]
    mean /= n
    s = 0.0
    for i in range(n):
        diff = x[i] - mean
        s += diff * diff
    return np.sqrt(s / (n - 1))


@njit
def _get_simplified_RS_random_walk(series):
    """
    Fonction from the hurst library (https://github.com/Mottl/hurst/)
    of Dmitry A. Mottl (pimped using numba).

    Copyright (c) 2017 Dmitry A. Mottl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    n = len(series)
    incs = np.empty(n - 1)
    for i in range(n - 1):
        incs[i] = series[i + 1] - series[i]
    R = np.max(series) - np.min(series)
    S = _std_numba(incs)
    if R == 0.0 or S == 0.0:
        return 0.0
    return R / S


@njit
def _get_simplified_RS_price(series):
    """
    Fonction from the hurst library (https://github.com/Mottl/hurst/)
    of Dmitry A. Mottl (pimped using numba).

    Copyright (c) 2017 Dmitry A. Mottl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    n = len(series)
    pcts = np.empty(n - 1)
    for i in range(n - 1):
        pcts[i] = series[i + 1] / series[i] - 1.0
    R = np.max(series) / np.min(series) - 1.0
    S = _std_numba(pcts)
    if R == 0.0 or S == 0.0:
        return 0.0
    return R / S


@njit
def _get_simplified_RS_change(series):
    """
    Fonction from the hurst library (https://github.com/Mottl/hurst/)
    of Dmitry A. Mottl (pimped using numba).

    Copyright (c) 2017 Dmitry A. Mottl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    n = len(series)
    _series = np.empty(n + 1)
    _series[0] = 0.0
    for i in range(1, n + 1):
        _series[i] = _series[i - 1] + series[i - 1]
    R = np.max(_series) - np.min(_series)
    S = _std_numba(series)
    if R == 0.0 or S == 0.0:
        return 0.0
    return R / S


@njit
def _compute_average_RS(series, w, mode):
    """
    Fonction from the hurst library (https://github.com/Mottl/hurst/)
    of Dmitry A. Mottl (pimped using numba).

    Copyright (c) 2017 Dmitry A. Mottl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    n = len(series)
    total = 0.0
    count = 0
    for start in range(0, n, w):
        if start + w > n:
            break
        window = series[start:start + w]
        rs = 0.0
        if mode == 0:
            rs = _get_simplified_RS_random_walk(window)
        elif mode == 1:
            rs = _get_simplified_RS_price(window)
        elif mode == 2:
            rs = _get_simplified_RS_change(window)
        if rs != 0.0:
            total += rs
            count += 1
    if count == 0:
        return 0.0
    return total / count


def _compute_Hc(series, kind="random_walk", min_window=10, max_window=None, simplified=True, min_sample=100):
    """
    Compute the Hurst exponent H and constant c from the Hurst equation:
        E(R/S) = c * T^H
    using the (simplified) rescaled range (RS) analysis.
    This optimized version uses Numba for accelerating the inner loops.

    Parameters
    ----------
    series : array-like
        Input time series data.
    kind : str, optional
        Type of series: 'random_walk', 'price' or 'change' (default is 'random_walk').
    min_window : int, optional
        Minimal window size for RS calculation (default is 10).
    max_window : int, optional
        Maximal window size for RS calculation (default is len(series)-1).
    simplified : bool, optional
        Use the simplified RS calculation (default True).
    min_sample : int, optional
        Minimum required length of series (default is 100).

    Returns
    -------
    tuple
        (H, c, [window_sizes, RS_values])
        where H is the Hurst exponent, c is the constant, and the last element contains
        the list of window sizes and corresponding average RS values (for further plotting).


    ------
    Fonction from the hurst library (https://github.com/Mottl/hurst/)
    of Dmitry A. Mottl (pimped using numba).

    Copyright (c) 2017 Dmitry A. Mottl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    """
    if len(series) < min_sample:
        raise ValueError(f"Series length must be >= min_sample={min_sample}")

    # Convert series to numpy array if needed
    if not isinstance(series, np.ndarray):
        series = np.array(series)
    if np.isnan(np.min(series)):
        raise ValueError("Series contains NaNs")

    # Determine mode for RS calculation based on kind
    if kind == 'random_walk':
        mode = 0
    elif kind == 'price':
        mode = 1
    elif kind == 'change':
        mode = 2
    else:
        raise ValueError("Unknown kind. Valid options are 'random_walk', 'price', 'change'.")

    max_window = max_window or (len(series) - 1)
    # Create a list of window sizes as powers of 10 with a step of 0.25 in log scale
    log_min = math.log10(min_window)
    log_max = math.log10(max_window)
    window_sizes = [int(10 ** x) for x in np.arange(log_min, log_max, 0.25)]
    window_sizes.append(len(series))

    RS_values = []
    for w in window_sizes:
        rs_avg = _compute_average_RS(series, w, mode)
        RS_values.append(rs_avg)

    # Prepare the design matrix for least squares regression:
    # log10(RS) = log10(c) + H * log10(window_size)
    A = np.vstack([np.log10(np.array(window_sizes)), np.ones(len(RS_values))]).T
    b = np.log10(np.array(RS_values))
    # Solve the least squares problem (this part remains in pure Python)
    H, c = np.linalg.lstsq(A, b, rcond=None)[0]
    c = 10 ** c
    return H, c, [window_sizes, RS_values]


def _hurst_exponent(series):
    """
    Calculates the Hurst exponent of a time series, which is a measure of the
    long-term memory of time series data.

    Parameters:
    -----------
    series : pandas.Series
        The input time series for which the Hurst exponent is to be calculated.

    Returns:
    --------
    float
        The Hurst exponent value. Returns NaN if the calculation fails.
    """

    try:
        H, c, data = _compute_Hc(series, kind='price')
    except:
        H = np.nan
    return H


def hurst(df: pd.DataFrame, col: str, window_size: int = 100) -> pd.DataFrame:
    """
    Compute the rolling Hurst exponent for a given column in a DataFrame.

    The Hurst exponent is a measure of the **long-term memory** of a time series.
    It helps determine whether a series is **mean-reverting**, **random**, or **trending**.

    Interpretation:
    - **H < 0.5**: Mean-reverting (e.g., stationary processes)
    - **H ≈ 0.5**: Random walk (e.g., Brownian motion)
    - **H > 0.5**: Trending behavior

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series data.
    col : str
        Column name on which the Hurst exponent is calculated.
    window_size : int, optional
        Rolling window size for the Hurst exponent computation (default = 100).

    Returns
    -------
    pd.Series
        A Series containing the rolling Hurst exponent values over the given window.
    """
    df_copy = df.copy()

    # Compute the rolling Hurst exponent using a helper function
    df_copy[f"hurst_{window_size}"] = df_copy[col].rolling(window=window_size, min_periods=window_size) \
        .apply(_hurst_exponent, raw=False)

    return df_copy[f"hurst_{window_size}"]
