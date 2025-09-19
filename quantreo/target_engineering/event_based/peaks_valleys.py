import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def detect_peaks_valleys(df: pd.DataFrame, col: str = 'close', **kwargs) -> pd.Series:
    """
    Detect peaks and valleys in a time series using scipy's find_peaks.

    This function labels turning points in a price series:
    - 1 for local maxima (**peaks**),
    - -1 for local minima (**valleys**),
    - 0 for all other points.

    It internally uses `scipy.signal.find_peaks` for both peak and valley detection.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the price data.
    col : str, optional
        The column name of the series to analyze (default is 'close').
    **kwargs :
        Additional keyword arguments passed directly to scipy.signal.find_peaks
        (e.g., distance=5, prominence=0.5, wlen=20, height=1.0, etc.)

    Returns
    -------
    pd.Series
        A Series of labels with the same index as `df`:
        - 1 for peaks,
        - -1 for valleys,
        - 0 for neutral points.

    Raises
    ------
    ValueError
        If the specified column is not found in the DataFrame.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    df = df.copy()
    prices = df[col].values

    # Peak detection
    peaks, _ = find_peaks(prices, **kwargs)
    valleys, _ = find_peaks(-prices, **kwargs)

    # Initialize columns
    df['peak'] = np.nan
    df['valley'] = np.nan
    df['label'] = 0

    # Assign peaks and valleys
    df.iloc[peaks, df.columns.get_loc('peak')] = df.iloc[peaks][col]
    df.iloc[valleys, df.columns.get_loc('valley')] = df.iloc[valleys][col]
    df.iloc[peaks, df.columns.get_loc('label')] = 1
    df.iloc[valleys, df.columns.get_loc('label')] = -1

    return df["label"]