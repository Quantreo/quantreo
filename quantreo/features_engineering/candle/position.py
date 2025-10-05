import pandas as pd


def internal_bar_strength(
    df: pd.DataFrame, high_col: str = "high", low_col: str = "low", close_col: str = "close"
) -> pd.Series:
    """
    Compute the Internal Bar Strength (IBS) indicator.

    The IBS is defined as:
        IBS = (Close - Low) / (High - Low)

    It measures where the closing price is located within the day's range,
    and is commonly used to detect short-term overbought or oversold conditions.

    Args:
        df (pd.DataFrame): DataFrame containing OHLC data.
        high_col (str): Name of the column representing the high price.
        low_col (str): Name of the column representing the low price.
        close_col (str): Name of the column representing the closing price.

    Returns:
        pd.Series: A Series with the IBS values, indexed like the input DataFrame.
    """
    range_ = df[high_col] - df[low_col]
    ibs = (df[close_col] - df[low_col]) / range_
    ibs = ibs.fillna(0.5)
    ibs.name = "IBS"
    return ibs
