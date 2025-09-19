import numpy as np
import pandas as pd


def close_to_close_volatility(df: pd.DataFrame, close_col: str = 'close', window_size: int = 30) -> pd.Series:
    """
    Calculate the rolling close-to-close volatility.md using standard deviation.

    This method computes the rolling standard deviation of the log returns,
    which represents the close-to-close volatility.md over a specified window.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the price data.
    window_size : int, optional
        The number of periods to include in the rolling calculation (default is 30).
    close_col : str, optional
        Column name for the closing prices (default is 'close').

    Returns
    -------
    volatility_series : pd.Series
        A Series indexed the same as `df`, containing the rolling close-to-close volatility.md.
        The first `window_size` rows will be NaN because there is insufficient data
        to compute the volatility.md in those windows.
    """
    # Ensure the required column exists
    if close_col not in df.columns:
        raise ValueError(f"The required column '{close_col}' is not present in the DataFrame.")

    # Compute log returns
    log_returns = df[close_col].pct_change().apply(lambda x: np.log(1 + x))

    # Compute rolling standard deviation of log returns
    volatility_series = log_returns.rolling(window=window_size, min_periods=window_size).std()

    # Name the resulting column
    volatility_series.name = "close_to_close_vol"

    return volatility_series