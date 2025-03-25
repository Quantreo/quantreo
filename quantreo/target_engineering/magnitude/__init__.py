import numpy as np
from quantreo.features_engineering.volatility import *


def future_returns(df, close_col='close', window_size=10, log_return=True):
    """
    Compute future returns over a specified window size.

    This function calculates the forward return for each observation
    over a given window_size, either in log-return or simple return format,
    using the specified close price column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing price data.
    close_col : str, optional (default='close')
        Name of the column to use as the close price.
    window_size : int
        Number of periods to shift forward to calculate the future return.
        This value is consistent with other Quantreo modules using the window_size parameter.
    log_return : bool, optional (default=True)
        If True, computes the log-return:
            log(close_t+window_size) - log(close_t)
        If False, computes the simple return:
            close_t+window_size / close_t - 1

    Returns
    -------
    pandas.Series
        A pandas Series containing the future returns (log or simple) for each row in the input DataFrame.
        The result will have NaN values for the last `window_size` rows due to the forward shift.

    Notes
    -----
    This target is part of the "Magnitude Targets" family within the Quantreo Target Engineering package.
    It is commonly used for regression models aimed at predicting return amplitude rather than direction.

    Examples
    --------
    >>> df = pd.DataFrame({'my_close': [100, 102, 101, 105, 110]})
    >>> future_returns(df, close_col='my_close', window_size=2, log_return=False)
    0    0.010000
    1    0.029412
    2    0.089109
    3         NaN
    4         NaN
    Name: fut_ret, dtype: float64
    """

    df_copy = df.copy()

    if log_return:
        df_copy["log_close"] = np.log(df_copy[close_col])
        df_copy["fut_ret"] = df_copy["log_close"].shift(-window_size) - df_copy["log_close"]
    else:
        df_copy["fut_ret"] = df_copy[close_col].shift(-window_size) / df_copy[close_col] - 1

    return df_copy["fut_ret"]


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
