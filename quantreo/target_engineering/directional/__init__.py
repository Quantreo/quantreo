import numpy as np
import pandas as pd
from ..magnitude import future_returns, continuous_barrier_labeling


def future_returns_sign(df: pd.DataFrame, close_col: str = 'close', window_size: int = 10, log_return: bool = True,
                        positive_label: int = 1, negative_label: int = 0) -> pd.Series:

    """
    Generate a directional target by computing future returns and binarizing them.

    This function internally calls `future_returns()` to compute the forward return,
    and then converts it into a binary directional target.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing price data.
    close_col : str, optional (default='close')
        Name of the column to use as the close price.
    window_size : int, optional (default=10)
        Number of periods to shift forward to calculate the future return.
    log_return : bool, optional (default=True)
        If True, computes the log-return, else simple return.
    positive_label : int, optional (default=1)
        Value assigned when the future return is strictly positive.
    negative_label : int, optional (default=0)
        Value assigned when the future return is zero or negative.

    Returns
    -------
    pandas.Series
        A pandas Series containing binary directional labels (1/0 or custom values).

    Notes
    -----
    This method is part of the "Directional Targets" family within the Quantreo Target Engineering package.

    Examples
    --------
    >>> df = pd.DataFrame({'close': [100, 102, 101, 105, 110]})
    >>> directional_target_from_returns(df, window_size=2, log_return=False)
    0    1
    1    1
    2    1
    3    0
    4    0
    dtype: int64
    """
    fut_ret = future_returns(df, close_col=close_col, window_size=window_size, log_return=log_return)
    labels = np.where(fut_ret > 0, positive_label, negative_label)
    return pd.Series(labels, index=fut_ret.index)


def quantile_label(df: pd.DataFrame, col: str, upper_quantile_level: float = 0.67,
                   lower_quantile_level: float | None = None, q_high: float | None = None, q_low: float | None = None,
                   return_thresholds: bool = False, positive_label: int = 1, negative_label: int = -1,
                   neutral_label: int = 0) -> pd.Series | tuple[pd.Series, float, float]:

    """
    Generate quantile-based labels (custom values for upper, lower, and neutral) and optionally return thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the target column (e.g., 'fut_ret').
    col : str
        Name of the column used for quantiles (e.g., 'fut_ret').
    upper_quantile_level : float, optional (default=0.67)
        The quantile level for the upper threshold.
    lower_quantile_level : float or None, optional (default=None)
        The quantile level for the lower threshold.
        If None, defaults to `1 - upper_quantile_level`.
    q_high : float or None, optional (default=None)
        Pre-calculated upper quantile value.
    q_low : float or None, optional (default=None)
        Pre-calculated lower quantile value.
    return_thresholds : bool, optional (default=False)
        If True, returns both the labels and the thresholds.
    positive_label : int or any, optional (default=1)
        Value assigned when the value is above the upper quantile.
    negative_label : int or any, optional (default=-1)
        Value assigned when the value is below the lower quantile.
    neutral_label : int or any, optional (default=0)
        Value assigned when the value is between the two quantiles.

    Returns
    -------
    labels : pandas.Series
        Series of custom labels.
    q_high : float (optional)
        Upper quantile value (if return_thresholds is True).
    q_low : float (optional)
        Lower quantile value (if return_thresholds is True).
    """

    if lower_quantile_level is None:
        lower_quantile_level = 1 - upper_quantile_level

    if q_high is None:
        q_high = df[col].quantile(upper_quantile_level)
    if q_low is None:
        q_low = df[col].quantile(lower_quantile_level)

    if q_high <= q_low:
        raise ValueError("Invalid quantiles: q_high must be strictly greater than q_low.")

    labels = pd.Series(neutral_label, index=df.index)
    labels.loc[df[col] > q_high] = positive_label
    labels.loc[df[col] < q_low] = negative_label

    if return_thresholds:
        return labels, q_high, q_low
    else:
        return labels


def double_barrier_labeling(df: pd.DataFrame, open_col: str = "open", high_col: str = "high", low_col: str = "low",
                            high_time_col: str = "high_time", low_time_col: str = "low_time", tp: float = 0.015,
                            sl: float = -0.015, buy: bool = True) -> pd.Series:
    """
    Compute double barrier classification labels based on TP/SL logic.

    This function wraps `continuous_barrier_labeling` and converts the continuous
    duration-based output into discrete labels:
        - 1  → Take Profit was hit first
        - -1 → Stop Loss was hit first
        - 0  → No barrier hit within max horizon

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with price and time columns.
    open_col, high_col, low_col : str
        Column names for OHLC prices.
    high_time_col, low_time_col : str
        Timestamps corresponding to high and low extremes.
    tp : float, optional
        Take Profit threshold.
    sl : float, optional
        Stop Loss threshold.
    buy : bool, optional
        Whether to simulate a long position.

    Returns
    -------
    pandas.Series
        A Series containing discrete labels: 1 (TP), -1 (SL), or 0 (none).
    """
    continuous = continuous_barrier_labeling(df, open_col=open_col, high_col=high_col, low_col=low_col,
                                             high_time_col=high_time_col, low_time_col=low_time_col, tp=tp, sl=sl,
                                             buy=buy)

    labels = continuous.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    labels.name = "barrier_label"
    return labels


def triple_barrier_labeling(df: pd.DataFrame, max_duration_h: float, open_col: str = "open", high_col: str = "high",
                            low_col: str = "low", high_time_col: str = "high_time", low_time_col: str = "low_time",
                            tp: float = 0.015, sl: float = -0.015, buy: bool = True) -> pd.Series:
    """
    Compute triple barrier classification labels based on TP/SL and a max holding time.

    Converts the continuous output of `continuous_barrier_labeling` into:
        -  1 → TP hit within max_duration_h
        - -1 → SL hit within max_duration_h
        -  0 → Timeout (barrier not reached in time)

    Parameters
    ----------
    df : pd.DataFrame
        Input price DataFrame.
    max_duration_h : float
        Maximum duration allowed (in hours) to reach TP or SL.
    open_col, high_col, low_col : str
        OHLC column names.
    high_time_col, low_time_col : str
        Timestamp columns for high and low extremes.
    tp : float
        Take Profit threshold.
    sl : float
        Stop Loss threshold.
    buy : bool
        Whether to simulate a long (True) or short (False) position.

    Returns
    -------
    pandas.Series
        A Series of labels: 1 (TP), -1 (SL), or 0 (neither hit within time).
    """
    durations = continuous_barrier_labeling(df, open_col=open_col, high_col=high_col, low_col=low_col,
        high_time_col=high_time_col, low_time_col=low_time_col, tp=tp, sl=sl, buy=buy)

    def label_fn(x):
        if abs(x) > max_duration_h:
            return 0
        return 1 if x > 0 else -1 if x < 0 else 0

    labels = durations.apply(label_fn)
    labels.name = "triple_barrier_label"
    return labels