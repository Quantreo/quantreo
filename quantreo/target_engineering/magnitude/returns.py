import numpy as np


def future_returns(df, close_col="close", window_size=10, log_return=True):
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
    """

    df_copy = df.copy()

    if log_return:
        df_copy["log_close"] = np.log(df_copy[close_col])
        df_copy["fut_ret"] = df_copy["log_close"].shift(-window_size) - df_copy["log_close"]
    else:
        df_copy["fut_ret"] = df_copy[close_col].shift(-window_size) / df_copy[close_col] - 1

    return df_copy["fut_ret"]
