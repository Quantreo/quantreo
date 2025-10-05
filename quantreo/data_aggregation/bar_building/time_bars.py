import pandas as pd
import numpy as np
from numba import njit
from typing import Callable, List, Tuple


@njit
def _build_time_bars(prices, volumes, timestamps_ns, window_ns):
    start_ts = timestamps_ns[0] // window_ns * window_ns
    end_ts = timestamps_ns[-1] // window_ns * window_ns + window_ns
    n_bins = (end_ts - start_ts) // window_ns

    bar_open = np.empty(n_bins, dtype=np.float64)
    bar_high = np.full(n_bins, -np.inf, dtype=np.float64)
    bar_low = np.full(n_bins, np.inf, dtype=np.float64)
    bar_close = np.empty(n_bins, dtype=np.float64)
    bar_volume = np.zeros(n_bins, dtype=np.float64)
    bar_count = np.zeros(n_bins, dtype=np.int64)
    bar_time = np.arange(start_ts, end_ts, window_ns)
    bar_start_idx = np.full(n_bins, -1, dtype=np.int64)
    bar_end_idx = np.full(n_bins, -1, dtype=np.int64)

    high_time = np.zeros(n_bins, dtype=np.int64)
    low_time = np.zeros(n_bins, dtype=np.int64)

    for i in range(len(timestamps_ns)):
        ts = timestamps_ns[i]
        price = prices[i]
        volume = volumes[i]
        bin_idx = (ts - start_ts) // window_ns

        if bar_count[bin_idx] == 0:
            bar_open[bin_idx] = price
            bar_start_idx[bin_idx] = i
            high_time[bin_idx] = ts
            low_time[bin_idx] = ts

        if price > bar_high[bin_idx]:
            bar_high[bin_idx] = price
            high_time[bin_idx] = ts
        if price < bar_low[bin_idx]:
            bar_low[bin_idx] = price
            low_time[bin_idx] = ts

        bar_close[bin_idx] = price
        bar_volume[bin_idx] += volume
        bar_count[bin_idx] += 1
        bar_end_idx[bin_idx] = i + 1

    valid = bar_count > 0
    return (
        bar_time[valid],
        bar_open[valid],
        bar_high[valid],
        bar_low[valid],
        bar_close[valid],
        bar_volume[valid],
        bar_count[valid],
        bar_start_idx[valid],
        bar_end_idx[valid],
        high_time[valid],
        low_time[valid],
    )


def ticks_to_time_bars(
    df: pd.DataFrame,
    resample_factor: str = "60min",
    col_price: str = "price",
    col_volume: str = "volume",
    additional_metrics: List[Tuple[Callable, str, List[str]]] = [],
) -> pd.DataFrame:
    """
    Convert tick-level data into fixed time bars using Numba, with optional additional metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by datetime, containing at least price and volume columns.
    col_price : str
        Name of the column containing tick prices.
    col_volume : str
        Name of the column containing tick volumes.
    resample_factor : str
        Resampling frequency (e.g., "1min", "5min", "1H", "1D").
    additional_metrics : List of (function, source, col_names)
        Each element is a tuple of:
        - a function applied to slices of data (must return float or tuple of floats),
        - the source: "price", "volume", or "price_volume",
        - a list of column names for the output(s) of the function.

    Returns
    -------
    pd.DataFrame
        Time bars indexed by period start time with OHLCV, tick count, and any custom metrics.
    """
    prices = df[col_price].to_numpy(np.float64)
    volumes = df[col_volume].to_numpy(np.float64)
    timestamps_ns = df.index.values.astype(np.int64)
    window_ns = pd.to_timedelta(resample_factor).value

    # Call numba-accelerated function
    times, opens, highs, lows, closes, vols, counts, start_idxs, end_idxs, high_times, low_times = (
        _build_time_bars(prices, volumes, timestamps_ns, window_ns)
    )

    # Base output dictionary
    out = {
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
        "number_ticks": counts,
        "high_time": pd.to_datetime(high_times),
        "low_time": pd.to_datetime(low_times),
    }

    # Compute additional metrics
    for func, source, col_names in additional_metrics:
        if source == "price":
            data = [func(prices[start:end]) for start, end in zip(start_idxs, end_idxs)]
        elif source == "volume":
            data = [func(volumes[start:end]) for start, end in zip(start_idxs, end_idxs)]
        elif source == "price_volume":
            data = [
                func(prices[start:end], volumes[start:end])
                for start, end in zip(start_idxs, end_idxs)
            ]
        else:
            raise ValueError(
                f"Invalid source '{source}'. Must be 'price', 'volume', or 'price_volume'."
            )

        if isinstance(data[0], tuple):
            for i, name in enumerate(col_names):
                out[name] = [row[i] for row in data]
        else:
            out[col_names[0]] = data

    df_out = pd.DataFrame(out, index=pd.to_datetime(times))
    df_out.index.name = "time"
    return df_out
