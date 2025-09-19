import numpy as np
from numba import njit
import pandas as pd
from scipy.stats import chi2
from scipy.stats import shapiro


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