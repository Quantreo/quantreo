import pandas as pd

def derivatives(df, col):
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
