import numpy as np
from typing import Tuple
import pandas as pd


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