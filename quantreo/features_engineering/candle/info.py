import pandas as pd
from typing import Tuple
import numpy as np

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