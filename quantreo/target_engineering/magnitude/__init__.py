import numpy as np
from quantreo.features_engineering.volatility import *
from tqdm import tqdm
from numba import njit


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


@njit
def _fast_barrier_buy(i: int, open_arr: np.ndarray, high_arr: np.ndarray, low_arr: np.ndarray,
                      high_time_arr: np.ndarray,
                      low_time_arr: np.ndarray, time_arr: np.ndarray, tp: float = 0.015, sl: float = -0.015) -> float:
    n = len(open_arr)
    for j in range(n):
        idx = i + j
        if idx >= n:
            break  # Avoid out-of-bounds

        open_price = open_arr[i]
        high_price = high_arr[i + j]
        low_price = low_arr[i + j]

        var_high = (high_price - open_price) / open_price
        var_low = (low_price - open_price) / open_price

        if (tp <= var_high) and (var_low <= sl):
            if high_time_arr[i + j] <= low_time_arr[i + j]:
                delta = high_time_arr[i + j] - time_arr[i]
                return delta / 3600
            else:
                delta = low_time_arr[i + j] - time_arr[i]
                return -delta / 3600

        elif tp <= var_high:
            delta = high_time_arr[i + j] - time_arr[i]
            return delta / 3600

        elif var_low <= sl:
            delta = low_time_arr[i + j] - time_arr[i]
            return -delta / 3600

    return 0.0

@njit
def _fast_barrier_sell(i: int, open_arr: np.ndarray, high_arr: np.ndarray, low_arr: np.ndarray,
                       high_time_arr: np.ndarray, low_time_arr: np.ndarray, time_arr: np.ndarray,
                       tp: float = 0.015, sl: float = -0.015) -> float:
    n = len(open_arr)
    for j in range(n):
        idx = i + j
        if idx >= n:
            break  # Avoid out-of-bounds

        open_price = open_arr[i]
        high_price = high_arr[i + j]
        low_price = low_arr[i + j]

        var_high = (high_price - open_price) / open_price
        var_low = (low_price - open_price) / open_price

        if (tp <= -var_low) and (-var_high <= sl):
            if low_time_arr[i + j] <= high_time_arr[i + j]:
                delta = low_time_arr[i + j] - time_arr[i]
                return delta / 3600
            else:
                delta = high_time_arr[i + j] - time_arr[i]
                return -delta / 3600

        elif tp <= -var_low:
            delta = low_time_arr[i + j] - time_arr[i]
            return delta / 3600

        elif -var_high <= sl:
            delta = high_time_arr[i + j] - time_arr[i]
            return -delta / 3600

    return 0.0


def _fast_ind_barrier(i: int, open_arr: np.ndarray, high_arr: np.ndarray, low_arr: np.ndarray,
                      high_time_arr: np.ndarray, low_time_arr: np.ndarray, time_arr: np.ndarray,
                      tp: float = 0.015, sl: float = -0.015, buy: bool = True) -> float:
    if buy:
        return _fast_barrier_buy(i, open_arr, high_arr, low_arr, high_time_arr, low_time_arr, time_arr, tp, sl)
    else:
        return _fast_barrier_sell(i, open_arr, high_arr, low_arr, high_time_arr, low_time_arr, time_arr, tp, sl)


def continuous_barrier_labeling(df: pd.DataFrame, open_col: str = "open", high_col: str = "high", low_col: str = "low",
                 high_time_col: str = "high_time", low_time_col: str = "low_time", tp: float = 0.015,
                 sl: float = -0.015, buy: bool = True) -> pd.Series:
    """
    Compute the time (in hours) to hit either a Take Profit (TP) or Stop Loss (SL) level
    after entering a trade, using a fast Numba-accelerated barrier labeling method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the following columns:
        - Price columns: `open_col`, `high_col`, `low_col`
        - Timestamp columns: `open_time_col`, `high_time_col`, `low_time_col`
    open_col : str, optional
        Name of the column containing the open price (default is 'open').
    high_col : str, optional
        Name of the column containing the high price (default is 'high').
    low_col : str, optional
        Name of the column containing the low price (default is 'low').
    open_time_col : str, optional
        Column name for the timestamp of the opening candle (default is 'open_time').
    high_time_col : str, optional
        Column name for the timestamp when the high occurred (default is 'high_time').
    low_time_col : str, optional
        Column name for the timestamp when the low occurred (default is 'low_time').
    tp : float, optional
        Take Profit threshold, as a relative change from open price (default is 0.015).
    sl : float, optional
        Stop Loss threshold, as a relative change from open price (default is -0.015).
    buy : bool, optional
        Whether to simulate a long position (True) or short position (False). Default is True.

    Returns
    -------
    pandas.Series
        A Series containing the time (in hours) required to hit either the TP or SL barrier after trade entry:
        - Positive values: TP was hit first.
        - Negative values: SL was hit first.
        - Zero: no barrier hit within the window or data ran out.
        The result is shifted by one row to prevent look-ahead bias.
    """
    df_copy = df.copy()

    required_cols = [open_col, high_col, low_col, high_time_col, low_time_col]
    for col in required_cols:
        if col not in df_copy.columns:
            raise ValueError(f"Missing required column: '{col}' in DataFrame.")

    if tp <= 0:
        raise ValueError(f"Take Profit (tp) should be strictly positive. Got {tp}")

    if sl >= 0:
        raise ValueError(f"Stop Loss (sl) should be strictly negative. Got {sl}")

    if len(df_copy) < 2:
        raise ValueError("DataFrame is too short to compute barriers.")


    df_copy.index.name = "time"
    df_copy = df_copy.reset_index(drop=False)

    # Convert timestamps to UNIX seconds
    df_copy["time_int"] = pd.to_datetime(df_copy["time"]).astype("int64") // 1_000_000_000
    df_copy["high_time_int"] = pd.to_datetime(df_copy[high_time_col]).astype("int64") // 1_000_000_000
    df_copy["low_time_int"] = pd.to_datetime(df_copy[low_time_col]).astype("int64") // 1_000_000_000

    # Extract arrays
    open_arr = df_copy[open_col].values
    high_arr = df_copy[high_col].values
    low_arr = df_copy[low_col].values
    high_time_arr = df_copy["high_time_int"].values
    low_time_arr = df_copy["low_time_int"].values
    time_arr = df_copy["time_int"].values

    # Barrier loop
    results = []
    for i in tqdm(range(len(df_copy))):
        try:
            label = _fast_ind_barrier(i, open_arr, high_arr, low_arr,
                                      high_time_arr, low_time_arr, time_arr, tp=tp, sl=sl, buy=buy)
        except Exception as e:
            print(f"Error at index {i}: {e}")
            label = 0.0
        results.append(label)

    label = pd.Series(results, index=df.index).shift(-1)
    label.iloc[-1] = 0
    return label