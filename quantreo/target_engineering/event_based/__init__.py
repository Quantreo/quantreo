import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def detect_peaks_valleys(df: pd.DataFrame, col: str = 'close', distance: int = 5, prominence: float = 0.5) -> pd.Series:
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
    distance : int, optional
        Minimum number of samples between two peaks or valleys.
        This parameter is passed to `scipy.signal.find_peaks`.
        A higher value will ignore smaller fluctuations and detect only well-separated turning points.
    prominence : float, optional
        Required prominence of peaks/valleys.
        Also passed to `scipy.signal.find_peaks`.
        Prominence represents how much a peak stands out from the surrounding data (in the "col" input unit).
        A higher value filters out smaller movements and keeps only significant peaks/valleys.

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
    peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)

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