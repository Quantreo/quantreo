import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def detect_peaks_valleys(df: pd.DataFrame, col: str = 'close', distance: int = 5, prominence: float = 0.5) -> pd.Series:
    """
    Detect peaks and valleys in a time series using scipy's find_peaks.

    The function returns a label series with:
    - 1 = Peak (local maximum)
    - -1 = Valley (local minimum)
    - 0 = Neutral point

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the price data.
    col : str, optional
        The column name of the series to analyze (default is 'close').
    distance : int, optional
        Minimum number of samples between consecutive peaks or valleys (default is 5).
    prominence : float, optional
        Required prominence of peaks/valleys (default is 0.5, in the same unit as the series).

    Returns
    -------
    pd.Series
        A Series of labels where:
        - 1 marks a peak,
        - -1 marks a valley,
        - 0 marks neutral points.

    Raises
    ------
    ValueError
        If the provided `col` is not present in the DataFrame.
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


