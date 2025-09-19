import pandas as pd
from ..magnitude import continuous_barrier_labeling


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