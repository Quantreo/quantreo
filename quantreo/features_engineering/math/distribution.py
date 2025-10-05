import numpy as np
import pandas as pd


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


def tail_index(
    df: pd.DataFrame, col: str = "close", window_size: int = 250, k_ratio: float = 0.10
) -> pd.Series:
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
        xmin = x_tail.min()  # k‑th order statistic
        out[end - 1] = k / np.log(x_tail / xmin).sum()

    return pd.Series(out, index=df.index, name=f"hill_{col}")


def bimodality_coefficient(df: pd.DataFrame, col: str, window_size: int = 100) -> pd.Series:
    """
    Compute the rolling Bimodality Coefficient (BC).

    Formula
    -------
        BC = (γ^2 + 1) / (κ + 3*(n-1)^2 / ((n-2)*(n-3)))

    where
        γ = skewness
        κ = excess kurtosis (Pandas .kurt() convention: normal=0)
        n = window size

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the input column.
    col : str
        Column name to compute BC on (e.g. returns, volatility).
    window_size : int, default=100
        Rolling window size. Must be >= 50.

    Returns
    -------
    pd.Series
        Series of rolling Bimodality Coefficient values, aligned with `df.index`.

    Notes
    -----
    - BC > 0.55 suggests bimodality or multimodality.
    - Sensitive to sample size: avoid too small windows.
    - Default of 100 provides a robust estimation for daily or intraday data.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not np.issubdtype(df[col].dtype, np.number):
        raise ValueError(f"Column '{col}' must be numeric.")
    if not isinstance(window_size, int) or window_size < 50:
        raise ValueError("'window_size' must be an integer >= 50.")

    series = df[col]

    skew_series = series.rolling(window=window_size).skew()
    kurt_series = series.rolling(window=window_size).kurt()

    def _bc_func(s, k, n):
        if pd.isna(s) or pd.isna(k):
            return np.nan
        denom = k + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        return (s**2 + 1) / denom if denom != 0 else np.nan

    bc = pd.Series(
        [_bc_func(s, k, window_size) for s, k in zip(skew_series, kurt_series)],
        index=series.index,
        name=f"bimodality_coefficient_{window_size}",
    )

    return bc
