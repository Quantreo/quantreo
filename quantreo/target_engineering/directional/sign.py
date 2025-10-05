import numpy as np
import pandas as pd
from ..magnitude import future_returns


def future_returns_sign(
    df: pd.DataFrame,
    close_col: str = "close",
    window_size: int = 10,
    log_return: bool = True,
    positive_label: int = 1,
    negative_label: int = 0,
) -> pd.Series:
    """
    Generate a directional target by computing future returns and binarizing them.

    This function internally calls `future_returns()` to compute the forward return,
    and then converts it into a binary directional target.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing price data.
    close_col : str, optional (default='close')
        Name of the column to use as the close price.
    window_size : int, optional (default=10)
        Number of periods to shift forward to calculate the future return.
    log_return : bool, optional (default=True)
        If True, computes the log-return, else simple return.
    positive_label : int, optional (default=1)
        Value assigned when the future return is strictly positive.
    negative_label : int, optional (default=0)
        Value assigned when the future return is zero or negative.

    Returns
    -------
    pandas.Series
        A pandas Series containing binary directional labels (1/0 or custom values).

    Notes
    -----
    This method is part of the "Directional Targets" family within the Quantreo Target Engineering package.
    """

    if window_size <= 0:
        raise ValueError("window_size must be strictly positive.")

    fut_ret = future_returns(
        df, close_col=close_col, window_size=window_size, log_return=log_return
    )
    labels = np.where(fut_ret > 0, positive_label, negative_label)
    return pd.Series(labels, index=fut_ret.index)
