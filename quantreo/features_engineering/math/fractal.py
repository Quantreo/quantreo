import numpy as np
from numba import njit
import math
import pandas as pd
import antropy as ant


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


def detrended_fluctuation(df: pd.DataFrame, col: str = "close", window_size: int = 100) -> pd.Series:
    """
    Calculate the rolling Detrended Fluctuation Analysis (DFA) exponent of a time series.

    DFA measures long-term memory and fractal scaling in a time series,
    making it suitable for detecting persistence or anti-persistence in market regimes.

    This function applies DFA over a rolling window, producing a time-varying
    indicator of signal regularity and self-similarity.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series.
    col : str, default="close"
        The name of the column on which to compute the DFA exponent.
    window_size : int, default=100
        Size of the rolling window (must be >= 100).

    Returns
    -------
    pd.Series
        A Series containing the rolling DFA exponents.
        The first (window_size - 1) values will be NaN.

    Notes
    -----
    This function uses AntroPy's implementation of DFA.
    AntroPy is licensed under the BSD 3-Clause License.
    © 2018–2025 Raphael Vallat — https://github.com/raphaelvallat/antropy
    """
    if window_size < 100:
        raise ValueError("DFA requires window_size >= 100 for stable estimation.")

    return df[col].rolling(window=window_size).apply(
        lambda x: ant.detrended_fluctuation(x)
        if len(x) == window_size else pd.NA,
        raw=True)