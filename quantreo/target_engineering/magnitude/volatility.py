from quantreo.features_engineering.volatility import (close_to_close_volatility, parkinson_volatility,
                                                      rogers_satchell_volatility, yang_zhang_volatility)
import pandas as pd


def future_volatility(df: pd.DataFrame, method: str = 'close_to_close', window_size: int = 20,
                      shift_forward: bool = True, **kwargs) -> pd.Series:
    """
    Compute the volatility over the next 'future_window' periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing OHLC or close price data.
    method : str
        Volatility estimation method. Options: ['close_to_close', 'parkinson', 'rogers_satchell', 'yang_zhang'].
    window_size : int
        Number of periods ahead to estimate future volatility.
    shift_forward : bool
        If True, volatility will be shifted backward to align with the current timestamp.
    kwargs : dict
        Additional parameters to pass to volatility estimators (e.g., close_col, high_col...).

    Returns
    -------
    pd.Series
        Series of future volatility values aligned on the current timestamp.
    """

    df_copy = df.copy()

    # Compute volatility on future window (shifted window to look forward)
    if method == 'close_to_close':
        vol = close_to_close_volatility(df_copy.shift(-window_size), window_size=window_size, **kwargs)
    elif method == 'parkinson':
        vol = parkinson_volatility(df_copy.shift(-window_size), window_size=window_size, **kwargs)
    elif method == 'rogers_satchell':
        vol = rogers_satchell_volatility(df_copy.shift(-window_size), window_size=window_size, **kwargs)
    elif method == 'yang_zhang':
        vol = yang_zhang_volatility(df_copy.shift(-window_size), window_size=window_size, **kwargs)
    else:
        raise ValueError("Invalid method selected. Choose from ['close_to_close', 'parkinson', 'rogers_satchell', 'yang_zhang'].")

    vol.name = "future_volatility"

    # Align volatility to the current timestamp
    # Explanation:
    # The volatility calculated from t+1 to t+N will be positioned at t+N by rolling()
    # We shift it back by +N to align this future information with timestamp t.
    if shift_forward:
        vol = vol.shift(window_size)

    return vol