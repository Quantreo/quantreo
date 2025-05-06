import numpy as np
from numba import njit
from typing import Tuple
import math
import pandas as pd
from scipy.stats import norm, chi2
import antropy as ant
from scipy.stats import shapiro


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


@njit(cache=True, fastmath=True)
def _adf_stat(x: np.ndarray, k: int, regression: str = "c") -> float:
    dx = np.diff(x)
    y = dx[k:]
    xlag = x[k:-1]
    n = y.size

    if regression == "c":
        p = 2 + k  # constant + lag + k diffs
        X = np.empty((n, p))
        X[:, 0] = 1.0
        X[:, 1] = xlag
        for j in range(k):
            X[:, 2 + j] = dx[k - (j + 1): -(j + 1) or None]
        target_idx = 1

    elif regression == "ct":
        p = 3 + k  # constant + trend + lag + k diffs
        X = np.empty((n, p))
        X[:, 0] = 1.0
        X[:, 1] = np.arange(k, len(x) - 1)
        X[:, 2] = xlag
        for j in range(k):
            X[:, 3 + j] = dx[k - (j + 1): -(j + 1) or None]
        target_idx = 2

    else:
        raise NotImplementedError("Only 'c' and 'ct' regressions are supported.")  # Unsupported regression mode

    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    sigma2 = np.dot(resid, resid) / (n - p)
    se_b = np.sqrt(sigma2 * np.linalg.inv(XtX)[target_idx, target_idx])
    return beta[target_idx] / se_b


def _adf_stat_wrapper(x: np.ndarray, k: int, regression: str = "c") -> float:
    try:
        return _adf_stat(np.array(x), k=k, regression=regression)
    except:
        return np.nan


def _adf_stat_to_pvalue(stat: float, regression: str = "c") -> float:
    stats_known = np.arange(-6, 3, 0.2)

    if regression == "c":
        pvalues_known = np.array([1.66612048e-07, 4.65495347e-07, 1.27117171e-06, 3.38720389e-06,
           8.79208358e-06, 2.21931547e-05, 5.43859357e-05, 1.29169640e-04,
           2.96832622e-04, 6.58900206e-04, 1.41051125e-03, 2.90731499e-03,
           5.76102775e-03, 1.09588716e-02, 1.99846792e-02, 3.48944003e-02,
           5.82737681e-02, 9.29972674e-02, 1.41736409e-01, 2.06245457e-01,
           2.86573099e-01, 3.80461694e-01, 4.83593470e-01, 5.82276119e-01,
           6.73595712e-01, 7.53264301e-01, 8.19122068e-01, 8.70982027e-01,
           9.10098777e-01, 9.38521617e-01, 9.58532086e-01, 9.72261594e-01,
           9.81494942e-01, 9.87615629e-01, 9.91636168e-01, 9.94265949e-01,
           9.95985883e-01, 9.97114142e-01, 9.97857576e-01, 9.98349017e-01,
           9.98672951e-01, 9.98882505e-01, 9.99010310e-01, 9.99075155e-01,
           1.00000000e+00])

    elif regression == "ct":
        pvalues_known = np.array([2.19685999e-06, 5.72200101e-06, 1.45728707e-05, 3.62096972e-05,
           8.75816742e-05, 2.05747283e-04, 4.68395913e-04, 1.03105554e-03,
           2.18970094e-03, 4.47697111e-03, 8.79370123e-03, 1.65606722e-02,
           2.98461511e-02, 5.13879267e-02, 8.44017024e-02, 1.32080985e-01,
           1.96944442e-01, 2.79892650e-01, 3.79618933e-01, 4.89850519e-01,
           6.01433772e-01, 7.04758323e-01, 7.92432292e-01, 8.60912560e-01,
           9.10502858e-01, 9.44114711e-01, 9.65682548e-01, 9.78951298e-01,
           9.86879034e-01, 9.91531946e-01, 9.94233168e-01, 9.95777535e-01,
           9.96616806e-01, 9.96986978e-01, 1.00000000e+00, 1.00000000e+00,
           1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
           1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
           1.00000000e+00])

    else:
        return np.nan

    if np.isnan(stat):
        return np.nan
    if stat <= stats_known[0]:
        return pvalues_known[0]
    elif stat >= stats_known[-1]:
        return pvalues_known[-1]
    return float(np.interp(stat, stats_known, pvalues_known))


def adf_test(df: pd.DataFrame, col: str, window_size: int, lags: int = None, regression: str = "c") -> tuple[pd.Series, pd.Series]:
    """
    Compute the Augmented Dickey-Fuller test in rolling windows to estimate stationarity over time.

    This function applies the ADF test in rolling fashion to a given column of a DataFrame.
    You can choose between a constant-only regression ('c') or a constant + linear trend ('ct').
    The p-values are approximated using fast interpolated tables, avoiding `statsmodels` overhead.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series to analyze.
    col : str
        Name of the column to test for stationarity.
    window_size : int
        Size of the rolling window to compute the ADF test.
    lags : int, optional (default=None)
        Number of lagged differences to include in the regression. If None, uses Schwert's rule.
    regression : str, optional (default='c')
        Type of regression to run:
        - 'c'  : constant only (tests stationarity around a non-zero mean)
        - 'ct' : constant + trend (tests stationarity around a linear trend)

    Returns
    -------
    tuple[pd.Series, pd.Series]
        - ADF statistic for each rolling window
        - Corresponding interpolated p-values
    """

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")

    if not np.issubdtype(df[col].dtype, np.number):
        raise ValueError(f"Column '{col}' must be numeric.")

    if not isinstance(window_size, int) or window_size < 10:
        raise ValueError(f"'window_size' must be a positive integer ≥ 10. Got {window_size}.")

    if lags is not None and (not isinstance(lags, int) or lags < 0):
        raise ValueError(f"'lags' must be None or a non-negative integer. Got {lags}.")

    if regression not in ["c", "ct"]:
        raise ValueError(f"'regression' must be either 'c' or 'ct'. Got '{regression}'.")

    series = df[col]

    if lags is None:
        k = int(np.floor(12 * (window_size / 100) ** 0.25))  # Schwert's rule
    else:
        k = lags

    stats = series.rolling(window=window_size).apply(
        lambda x: _adf_stat_wrapper(x, k=k, regression=regression),
        raw=True
    )


    p_val = stats.apply(lambda stat: _adf_stat_to_pvalue(stat, regression=regression))

    return stats.rename("adf_stat"), p_val.rename("adf_pval")



@njit(cache=True, fastmath=True)
def _arch_lm_only(y: np.ndarray, nlags: int, ddof: int = 0) -> float:
    nobs = y.shape[0] - nlags
    if nobs <= nlags + 1:
        return np.nan

    y_target = y[nlags:]
    y_lagged = np.empty((nobs, nlags))
    for i in range(nlags):
        y_lagged[:, i] = y[nlags - i - 1: -i - 1]

    X = np.ones((nobs, nlags + 1))
    X[:, 1:] = y_lagged

    XtX = X.T @ X
    Xty = X.T @ y_target
    beta = np.linalg.solve(XtX, Xty)
    y_hat = X @ beta
    resid = y_target - y_hat

    rss = np.dot(resid, resid)
    tss = np.dot(y_target - y_target.mean(), y_target - y_target.mean())
    r_squared = 1 - rss / tss

    return (nobs - ddof) * r_squared


def arch_test(df: pd.DataFrame, col: str, window_size: int = 60, lags: int = 5, ddof: int = 0) -> tuple[pd.Series, pd.Series]:
    """
    Compute the ARCH test (Engle) over rolling windows to detect conditional heteroskedasticity.

    This function applies the ARCH Lagrange Multiplier test in a rolling fashion
    to a given time series. It returns both the LM statistic and the associated p-value.
    The ARCH test measures whether volatility is autocorrelated (i.e., clustering),
    which is common in financial time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series data.
    col : str
        Name of the column to test (typically returns or residuals).
    window_size : int, optional (default=60)
        Size of the rolling window used to estimate ARCH effects.
    lags : int, optional (default=5)
        Number of lags to include in the ARCH regression (squared residuals).
    ddof : int, optional (default=0)
        Degrees of freedom adjustment (useful when residuals come from a fitted model).

    Returns
    -------
    arch_stat : pd.Series
        Rolling series of the LM statistics from the ARCH test.
    arch_pval : pd.Series
        Rolling series of the associated p-values (under Chi2 distribution).

    Raises
    ------
    ValueError
        If inputs are invalid: missing column, non-numeric data, or incorrect parameters.
    """

    # --- Input validation ---
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")

    if not np.issubdtype(df[col].dtype, np.number):
        raise ValueError(f"Column '{col}' must contain numeric values.")

    if not isinstance(window_size, int) or window_size <= 1:
        raise ValueError(f"'window_size' must be a positive integer > 1. Got {window_size}.")

    if lags is not None and (not isinstance(lags, int) or lags < 1):
        raise ValueError(f"'lags' must be an integer >= 1. Got {lags}.")

    if not isinstance(ddof, int) or ddof < 0:
        raise ValueError(f"'ddof' must be a non-negative integer. Got {ddof}.")

    # --- Determine nlags ---
    if lags is None:
        nlags = int(np.floor(12 * (window_size / 100) ** 0.25))  # Schwert's rule
    else:
        nlags = lags

    if window_size <= nlags + 1:
        raise ValueError(f"'window_size' must be greater than 'lags + 1' for regression to be valid.")

    # --- Rolling ARCH computation ---
    lm_stats = []
    index = df.index[window_size:]

    for i in range(window_size, len(df)):
        window_data = df[col].iloc[i - window_size:i].values
        y = window_data ** 2
        lm_stat = _arch_lm_only(y, nlags, ddof)
        lm_stats.append(lm_stat)

    lm_stats = np.array(lm_stats)
    lm_pvals = chi2.sf(lm_stats, df=nlags)

    return (
        pd.Series(lm_stats, index=index, name="arch_stat"),
        pd.Series(lm_pvals, index=index, name="arch_pval")
    )


def skewness(df: pd.DataFrame, col: str, window_size: int = 60) -> pd.Series:
    """
    Compute the skewness (third standardized moment) over a rolling window.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series data.
    col : str
        Name of the column to compute skewness on.
    window_size : int, optional (default=60)
        Number of periods for the rolling window.

    Returns
    -------
    pd.Series
        Rolling skewness of the specified column.

    Examples
    --------
    df["skew"] = rolling_skewness(df, col="returns", window_size=50)
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not np.issubdtype(df[col].dtype, np.number):
        raise ValueError(f"Column '{col}' must be numeric.")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f"'window_size' must be a positive integer. Got {window_size}.")

    return df[col].rolling(window=window_size).skew().rename("skewness")



def kurtosis(df: pd.DataFrame, col: str, window_size: int = 60) -> pd.Series:
    """
    Compute the kurtosis (fourth standardized moment) over a rolling window.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series data.
    col : str
        Name of the column to compute kurtosis on.
    window_size : int, optional (default=60)
        Number of periods for the rolling window.

    Returns
    -------
    pd.Series
        Rolling kurtosis of the specified column.

    Examples
    --------
    df["kurtosis"] = rolling_kurtosis(df, col="returns", window_size=50)
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not np.issubdtype(df[col].dtype, np.number):
        raise ValueError(f"Column '{col}' must be numeric.")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f"'window_size' must be a positive integer. Got {window_size}.")

    return df[col].rolling(window=window_size).kurt().rename("kurtosis")


def sample_entropy(df: pd.DataFrame, col: str = "close", window_size: int = 100, order: int = 2) -> pd.Series:
    """
    Calculate the rolling Sample Entropy of a time series.

    Sample Entropy quantifies the level of irregularity or unpredictability
    in a signal. It is often used to detect dynamic changes in structural
    complexity over time.

    This function applies Sample Entropy on a sliding window, allowing you
    to observe how entropy evolves across a time series.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series.
    col : str, default="close"
        The name of the column on which to compute the entropy.
    window_size : int, default=100
        Size of the rolling window (must be >= 10).
    order : int, default=2
        Embedding dimension used in the entropy calculation (must be >= 1).

    Returns
    -------
    pd.Series
        A Series containing the rolling Sample Entropy values. The first
        (window_size - 1) values will be NaN.

    Notes
    -----
    This function uses AntroPy's implementation of Sample Entropy.
    AntroPy is licensed under the BSD 3-Clause License.
    © 2018–2025 Raphael Vallat — https://github.com/raphaelvallat/antropy
    """
    if window_size < 10:
        raise ValueError("Sample entropy requires window_size >= 10.")

    if order < 1:
        raise ValueError("Parameter 'order' must be >= 1.")

    return df[col].rolling(window=window_size).apply(
        lambda x: ant.sample_entropy(x, order=order) if len(x) == window_size else np.nan,
        raw=True)


def spectral_entropy(df: pd.DataFrame, col: str = "close", window_size: int = 100, sf: int = 1,
                             method: str = 'welch', normalize: bool = True, nperseg: int = None) -> pd.Series:
    """
    Calculate the rolling Spectral Entropy of a time series.

    Spectral Entropy quantifies the flatness or complexity of the power
    spectral density of a signal. It provides insight into the frequency
    content and structure of a time series.

    This function applies spectral entropy over a rolling window, allowing
    dynamic tracking of complexity in the frequency domain.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series.
    col : str, default="close"
        The name of the column on which to compute the entropy.
    window_size : int, default=100
        Size of the rolling window (must be >= 16).
    sf : int, default=1
        Sampling frequency used in spectral estimation (must be > 0).
    method : str, default="welch"
        Method used to compute the power spectral density ("welch" or "fft").
    normalize : bool, default=True
        Whether to normalize entropy to [0, 1].
    nperseg : int, optional
        Segment length for Welch's method. If None, defaults to min(window_size, window_size // 2).

    Returns
    -------
    pd.Series
        A Series containing the rolling Spectral Entropy values. The first
        (window_size - 1) values will be NaN.

    Notes
    -----
    This function uses AntroPy's implementation of Spectral Entropy.
    AntroPy is licensed under the BSD 3-Clause License.
    © 2018–2025 Raphael Vallat — https://github.com/raphaelvallat/antropy
    """
    if window_size < 16:
        raise ValueError("Spectral entropy requires window_size >= 16 for stable estimation.")

    if sf <= 0:
        raise ValueError("Sampling frequency 'sf' must be strictly positive.")

    if method not in ["welch", "fft"]:
        raise ValueError("Method must be 'welch' or 'fft'.")

    def compute_entropy(x):
        local_nperseg = nperseg if nperseg is not None else min(window_size, window_size // 2)
        return ant.spectral_entropy(x, sf=sf, method=method, normalize=normalize, nperseg=local_nperseg)

    return df[col].rolling(window=window_size).apply(
        lambda x: compute_entropy(x) if len(x) == window_size else np.nan,
        raw=True)


def permutation_entropy(df: pd.DataFrame, col: str = "close", window_size: int = 100, order: int = 3,
                                delay: int = 1, normalize: bool = True) -> pd.Series:
    """
    Calculate the rolling Permutation Entropy of a time series.

    Permutation Entropy quantifies the complexity of temporal ordering in a signal.
    It is particularly useful for detecting subtle dynamic changes in structure.

    This function computes Permutation Entropy over a sliding window,
    providing a real-time view of signal complexity.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series.
    col : str, default="close"
        The name of the column on which to compute the entropy.
    window_size : int, default=100
        Size of the rolling window (must be >= 10).
    order : int, default=3
        Embedding dimension for permutation patterns (must be >= 2).
    delay : int, default=1
        Time delay between points used in embedding (must be >= 1).
    normalize : bool, default=True
        Whether to normalize entropy to [0, 1].

    Returns
    -------
    pd.Series
        A Series containing the rolling Permutation Entropy values.
        The first (window_size - 1) values will be NaN.

    Notes
    -----
    This function uses AntroPy's implementation of Permutation Entropy.
    AntroPy is licensed under the BSD 3-Clause License.
    © 2018–2025 Raphael Vallat — https://github.com/raphaelvallat/antropy
    """
    if window_size < 10:
        raise ValueError("Permutation entropy requires window_size >= 10.")

    if order < 2:
        raise ValueError("Embedding 'order' must be >= 2.")

    if delay < 1:
        raise ValueError("Delay must be >= 1.")

    return df[col].rolling(window=window_size).apply(
        lambda x: ant.perm_entropy(x, order=order, delay=delay, normalize=normalize)
        if len(x) == window_size else np.nan,
        raw=True)


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


def petrosian_fd(df: pd.DataFrame, col: str = "close", window_size: int = 100) -> pd.Series:
    """
    Calculate the rolling Petrosian Fractal Dimension (FD) of a time series.

    Petrosian FD measures the structural complexity of a signal based on
    changes in the direction of the signal's first derivative.

    This function applies the Petrosian FD over a rolling window,
    producing a time series that tracks signal complexity in real-time.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series.
    col : str, default="close"
        The name of the column on which to compute the fractal dimension.
    window_size : int, default=100
        Size of the rolling window (must be >= 10).

    Returns
    -------
    pd.Series
        A Series containing the rolling Petrosian FD values.
        The first (window_size - 1) values will be NaN.

    Notes
    -----
    This function uses AntroPy's implementation of Petrosian FD.
    AntroPy is licensed under the BSD 3-Clause License.
    © 2018–2025 Raphael Vallat — https://github.com/raphaelvallat/antropy
    """
    if window_size < 10:
        raise ValueError("Petrosian fractal dimension requires window_size >= 10.")

    return df[col].rolling(window=window_size).apply(
        lambda x: ant.petrosian_fd(x) if len(x) == window_size else np.nan,
        raw=True)


def tail_index(df: pd.DataFrame, col: str = "close", window_size: int = 250, k_ratio: float = 0.10) -> pd.Series:
    """
    Rolling Hill tail‑index (α̂, *without* the +1 bias‑correction).

    *Right‑tail* estimator – **`df[col]` must contain strictly positive values**
    (e.g. absolute returns, drawdown magnitudes).
    Any window that includes ≤ 0 is skipped.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame.
    col : str, default "close"
        Column on which to compute α̂(t).
    window_size : int, default 250
        Rolling window length *n*.
    k_ratio : float, default 0.10
        Fraction of the window regarded as the tail
        (`k = max(1, int(round(k_ratio * window_size)))`).
        5 – 15 % is a common compromise between bias and variance.

    Returns
    -------
    pd.Series
        α̂(t) aligned with `df.index`; the first `window_size−1` points are `NaN`.

    """
    if not 0 < k_ratio < 1:
        raise ValueError("k_ratio must lie in the interval (0, 1).")

    k = max(1, int(round(k_ratio * window_size)))
    if k >= window_size:
        raise ValueError("k_ratio * window_size must be < window_size.")

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in df.")

    ts = df[col].values.astype("float64") + 10e-10
    out = np.full_like(ts, np.nan)

    for end in range(window_size, len(ts) + 1):
        w = ts[end - window_size : end]

        # Require strictly positive values for right‑tail estimation
        if np.any(w <= 0.0):
            continue

        # k largest observations (O(n))
        x_tail = np.partition(w, window_size - k)[-k:]
        xmin = x_tail.min()                         # k‑th order statistic
        out[end - 1] = k / np.log(x_tail / xmin).sum()

    return pd.Series(out, index=df.index, name=f"hill_{col}")


def shapiro_wilk(df: pd.DataFrame, col: str, window_size: int) -> tuple[pd.Series, pd.Series]:
    """
    Rolling Shapiro-Wilk test for normality on a time series column.

    This function evaluates the null hypothesis that the data in the specified column
    comes from a normal distribution. It applies the test over a rolling window
    of fixed size and returns both the test statistic (W) and the associated p-value
    at each time step.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series.
    col : str
        Name of the column to test for normality.
    window_size : int
        Rolling window size.

    Returns
    -------
    stat_series : pd.Series
        Series of W statistics from the Shapiro-Wilk test.
    pval_series : pd.Series
        Series of p-values corresponding to each window.
    """
    values = df[col].values
    stat_results = []
    pval_results = []

    for i in range(window_size, len(values) + 1):
        window_data = values[i - window_size:i]
        if np.any(np.isnan(window_data)):
            stat_results.append(np.nan)
            pval_results.append(np.nan)
        else:
            w_stat, p_val = shapiro(window_data)
            stat_results.append(w_stat)
            pval_results.append(p_val)

    pad = [np.nan] * (window_size - 1)
    index = df.index
    return (pd.Series(pad + stat_results, index=index, name=f"{col}_shapiro_stat"),
        pd.Series(pad + pval_results, index=index, name=f"{col}_shapiro_pval"))