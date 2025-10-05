import numpy as np
from tqdm import tqdm
from numba import njit
import pandas as pd


@njit
def _fast_barrier_buy(
    i: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    high_time_arr: np.ndarray,
    low_time_arr: np.ndarray,
    time_arr: np.ndarray,
    tp: float = 0.015,
    sl: float = -0.015,
) -> float:
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
def _fast_barrier_sell(
    i: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    high_time_arr: np.ndarray,
    low_time_arr: np.ndarray,
    time_arr: np.ndarray,
    tp: float = 0.015,
    sl: float = -0.015,
) -> float:
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


def _fast_ind_barrier(
    i: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    high_time_arr: np.ndarray,
    low_time_arr: np.ndarray,
    time_arr: np.ndarray,
    tp: float = 0.015,
    sl: float = -0.015,
    buy: bool = True,
) -> float:
    if buy:
        return _fast_barrier_buy(
            i, open_arr, high_arr, low_arr, high_time_arr, low_time_arr, time_arr, tp, sl
        )
    else:
        return _fast_barrier_sell(
            i, open_arr, high_arr, low_arr, high_time_arr, low_time_arr, time_arr, tp, sl
        )


def continuous_barrier_labeling(
    df: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    high_time_col: str = "high_time",
    low_time_col: str = "low_time",
    tp: float = 0.015,
    sl: float = -0.015,
    buy: bool = True,
) -> pd.Series:
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
    df_copy["high_time_int"] = (
        pd.to_datetime(df_copy[high_time_col]).astype("int64") // 1_000_000_000
    )
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
            label = _fast_ind_barrier(
                i,
                open_arr,
                high_arr,
                low_arr,
                high_time_arr,
                low_time_arr,
                time_arr,
                tp=tp,
                sl=sl,
                buy=buy,
            )
        except Exception as e:
            print(f"Error at index {i}: {e}")
            label = 0.0
        results.append(label)

    label = pd.Series(results, index=df.index).shift(-1)
    label.iloc[-1] = 0
    return label
