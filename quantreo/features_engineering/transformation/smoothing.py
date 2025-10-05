import pandas as pd


def mma(df: pd.DataFrame, col: str, window_size: int = 3) -> pd.Series:
    """
    Calculate the Median Moving Average (MMA) using Pandas rolling.median.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the input data.
    col : str
        Name of the column on which to compute the MMA.
    window_size : int, optional
        The window size for computing the MMA (default is 3).

    Returns
    -------
    mma_series : pandas.Series
        A Series indexed the same as the input DataFrame, containing the MMA values.
        The first (window_size - 1) entries will be NaN due to insufficient data.
    """
    # Verify that the required column exists
    if col not in df.columns:
        raise ValueError(f"The column '{col}' is not present in the DataFrame.")

    # Compute the Median Moving Average using Pandas' rolling.median()
    mma_series = df[col].rolling(window=window_size).median()

    # Rename the series for clarity
    mma_series.name = f"mma_{window_size}"

    return mma_series
