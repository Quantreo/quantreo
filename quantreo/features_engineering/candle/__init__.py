import pandas as pd
import numpy as np
from typing import Tuple
from numba import njit


def candle_information(df: pd.DataFrame, open_col: str = 'open', high_col: str = 'high', low_col: str = 'low',
                       close_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute candle information indicators for a given OHLC DataFrame.

    This function calculates:
      - 'candle_way': Indicator for the candle's color (1 if close > open, -1 otherwise).
      - 'filling': The filling percentage, computed as the absolute difference between
                   close and open divided by the range (high - low).
      - 'amplitude': The candle amplitude as a percentage, calculated as the absolute difference
                     between close and open divided by the average of open and close, multiplied by 100.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing OHLC data.
    open_col : str, optional
        Column name for open prices (default is 'open').
    high_col : str, optional
        Column name for high prices (default is 'high').
    low_col : str, optional
        Column name for low prices (default is 'low').
    close_col : str, optional
        Column name for close prices (default is 'close').

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        - candle_way (pd.Series[int]): The direction of the candle (`1` for bullish, `-1` for bearish).
        - filling (pd.Series[float]): The proportion of the candle range occupied by the body.
        - amplitude (pd.Series[float]): The relative size of the candle in percentage.
    """

    df_copy = df.copy()

    # Candle color: 1 if close > open, else -1.
    df_copy["candle_way"] = -1
    df_copy.loc[df_copy[open_col] < df_copy[close_col], "candle_way"] = 1

    # Filling percentage: |close - open| / |high - low|
    df_copy["filling"] = np.abs(df_copy[close_col] - df_copy[open_col]) / np.abs(df_copy[high_col] - df_copy[low_col])

    # Amplitude: |close - open| / ((open + close)/2)
    df_copy["amplitude"] = (np.abs(df_copy[close_col] - df_copy[open_col]) / (
            (df_copy[open_col] + df_copy[close_col]) / 2))

    return df_copy["candle_way"], df_copy["filling"], df_copy["amplitude"]


def compute_spread(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low') -> pd.Series:
    """
    Compute the spread between the high and low price columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing price data.
    high_col : str, optional
        Column name for the high prices (default is 'high').
    low_col : str, optional
        Column name for the low prices (default is 'low').

    Returns
    -------
    spread_series : pandas.Series
        A Series indexed the same as `df`, containing the spread values.
    """
    # Check that the necessary columns exist in the DataFrame
    for col in [high_col, low_col]:
        if col not in df.columns:
            raise ValueError(f"The required column '{col}' is not present in the DataFrame.")

    # Compute the spread
    spread_series = df[high_col] - df[low_col]

    # Return as a Series with a clear name
    return pd.Series(spread_series, name="spread", index=df.index)


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



def price_distribution(df: pd.DataFrame, col: str, window_size: int = 60,
                       start_percentage: float = 0.25, end_percentage: float = 0.75) -> pd.Series:
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
    return df[col].rolling(window_size).apply(
        lambda x: _close_percentage_in_range(x, start_percentage, end_percentage),
        raw=True)


def internal_bar_strength(df: pd.DataFrame, high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pd.Series:
    """
    Compute the Internal Bar Strength (IBS) indicator.

    The IBS is defined as:
        IBS = (Close - Low) / (High - Low)

    It measures where the closing price is located within the day's range,
    and is commonly used to detect short-term overbought or oversold conditions.

    Args:
        df (pd.DataFrame): DataFrame containing OHLC data.
        high_col (str): Name of the column representing the high price.
        low_col (str): Name of the column representing the low price.
        close_col (str): Name of the column representing the closing price.

    Returns:
        pd.Series: A Series with the IBS values, indexed like the input DataFrame.
    """
    range_ = df[high_col] - df[low_col]
    ibs = (df[close_col] - df[low_col]) / range_
    ibs.name = "IBS"
    return ibs