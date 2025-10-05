import pandas as pd
import numpy as np


def fisher_transform(
    df: pd.DataFrame, high_col: str = "high", low_col: str = "low", window_size: int = 10
) -> pd.Series:
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

    if high_col not in df.columns or low_col not in df.columns:
        raise ValueError(f"Column '{high_col}' or '{low_col}' not found in DataFrame.")
    if not np.issubdtype(df[high_col].dtype, np.number) or not np.issubdtype(df[low_col].dtype, np.number):
        raise ValueError(f"Column '{high_col}' or '{low_col}' must be numeric.")

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


def logit_transform(df: pd.DataFrame, col: str, eps: float = 1e-6) -> pd.Series:
    """
    Apply the logit transformation to a column in a DataFrame.

    The logit function is defined as:
        logit(x) = log(x / (1 - x))

    To avoid infinities when x is close to 0 or 1, values are clipped to
    [eps, 1 - eps].

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data.
    col : str
        Column name on which to apply the transformation.
    eps : float, optional (default=1e-6)
        Small value used for clipping to prevent numerical issues at 0 or 1.

    Returns
    -------
    pd.Series
        Transformed series with the same index as the input DataFrame.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not np.issubdtype(df[col].dtype, np.number):
        raise ValueError(f"Column '{col}' must be numeric.")

    series = df[col].to_numpy(dtype=np.float64)
    # Clip to avoid division by zero or log(0)
    clipped = np.clip(series, eps, 1 - eps)

    transformed = np.log(clipped / (1.0 - clipped))

    return pd.Series(transformed, index=df.index, name=f"{col}_logit")


def neg_log_transform(df: pd.DataFrame, col: str, eps: float = 1e-6) -> pd.Series:
    """
    Apply the negative logarithm transformation to a column in a DataFrame.

    This transformation is defined as:
        f(x) = -log(x)

    It is commonly used to highlight small values (e.g., p-values).
    To avoid issues with log(0), values are clipped to [eps, 1].

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data.
    col : str
        Column name on which to apply the transformation.
    eps : float, optional (default=1e-12)
        Small positive value used for clipping to prevent log(0).

    Returns
    -------
    pd.Series
        Transformed series with the same index as the input DataFrame.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not np.issubdtype(df[col].dtype, np.number):
        raise ValueError(f"Column '{col}' must be numeric.")

    series = df[col].to_numpy(dtype=np.float64)
    # Clip values to avoid log(0)
    clipped = np.clip(series, eps, 1.0)

    transformed = -np.log(clipped)

    return pd.Series(transformed, index=df.index, name=f"{col}_neg_log")
