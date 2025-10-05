import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _generate_savgol_last_point_coeffs(window_size: int, polyorder: int) -> np.ndarray:
    x = np.arange(window_size)
    A = np.vander(x, N=polyorder + 1, increasing=True)
    AtA_inv = np.linalg.inv(A.T @ A)
    coeffs = A @ (AtA_inv @ A.T)
    return coeffs[-1]  # We extract weights for the last row (latest point)


def savgol_filter(
    df: pd.DataFrame, col: str = "close", window_size: int = 11, polyorder: int = 3
) -> pd.Series:
    """
    Compute a causal Savitzky-Golay filter using optimized matrix operations.

    This version reproduces the behavior of:
        df[col].rolling(window_size).apply(lambda x: savgol_filter(x, window_size, polyorder)[-1])

    It applies a rolling polynomial regression over a past-only window, avoiding look-ahead bias.

    Args:
        df (pd.DataFrame): DataFrame containing the input series.
        col (str): Column name of the series to smooth.
        window_size (int): Length of the rolling window (must be odd).
        polyorder (int): Degree of the fitted polynomial (must be < window_size).

    Returns:
        pd.Series: Smoothed series using the causal Savitzky-Golay filter.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")

    if polyorder >= window_size:
        raise ValueError("polyorder must be less than window_size.")

    series = df[col]
    coeffs = _generate_savgol_last_point_coeffs(window_size, polyorder)

    x = series.to_numpy(dtype=np.float64)
    result = np.full_like(x, np.nan)

    # Create rolling windows and apply dot product with precomputed coefficients
    X = sliding_window_view(x, window_shape=window_size)
    Y = X @ coeffs

    # Assign the result to the end of each window
    result[window_size - 1 :] = Y

    return pd.Series(result, index=series.index, name=f"{col}_savgol_causal")
