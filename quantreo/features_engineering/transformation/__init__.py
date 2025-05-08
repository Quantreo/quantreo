import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def fisher_transform(df: pd.DataFrame, high_col: str = "high", low_col: str = "low", window_size: int = 10) -> pd.Series:
    """
    Compute the Fisher Transform indicator.

    The Fisher Transform maps price data into a Gaussian-like distribution
    using the formula:
        Fisher = 0.5 * ln((1 + x) / (1 - x))

    Where:
        x = 2 * (median - min) / (max - min) - 1

    It is typically used to detect turning points and overbought/oversold conditions.

    Args:
        df (pd.DataFrame): DataFrame containing OHLC price data.
        high_col (str): Column name for the high price.
        low_col (str): Column name for the low price.
        window_size (int): Rolling window used to normalize the price range.

    Returns:
        pd.Series: A Series containing the Fisher Transform values.
    """
    # Compute the median price from high and low
    median_price = (df[high_col] + df[low_col]) / 2

    # Rolling min and max over the selected window
    min_roll = df[low_col].rolling(window=window_size).min()
    max_roll = df[high_col].rolling(window=window_size).max()

    # Normalize median price to [-1, 1]
    raw = 2 * (median_price - min_roll) / (max_roll - min_roll) - 1
    raw = raw.clip(lower=-0.999, upper=0.999)  # prevent log explosion

    # Apply the Fisher Transform
    fisher = 0.5 * np.log((1 + raw) / (1 - raw))

    return pd.Series(fisher, index=df.index, name="fisher")


def _generate_savgol_last_point_coeffs(window_size: int, polyorder: int) -> np.ndarray:
    x = np.arange(window_size)
    A = np.vander(x, N=polyorder + 1, increasing=True)
    AtA_inv = np.linalg.inv(A.T @ A)
    coeffs = A @ (AtA_inv @ A.T)
    return coeffs[-1]  # We extract weights for the last row (latest point)


def savgol_filter(df: pd.DataFrame, col: str = "close", window_size: int = 11, polyorder: int = 3) -> pd.Series:
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
    result[window_size - 1:] = Y

    return pd.Series(result, index=series.index, name=f"{col}_savgol_causal")
