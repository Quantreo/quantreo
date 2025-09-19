import pandas as pd
import numpy as np
from numba import njit
from typing import Callable, List, Tuple


@njit
def _build_tick_bars(prices, volumes, timestamps_ns, tick_per_bar):
    n_ticks = len(prices)
    n_bars = n_ticks // tick_per_bar

    bars = np.empty((n_bars, 10), dtype=np.float64)
    indices = np.empty((n_bars, 2), dtype=np.int64)

    for i in range(n_bars):
        start = i * tick_per_bar
        end = start + tick_per_bar

        p = prices[start:end]
        v = volumes[start:end]
        t = timestamps_ns[start:end]

        high_idx = np.argmax(p)
        low_idx = np.argmin(p)

        bars[i, 0] = t[0]
        bars[i, 1] = p[0]
        bars[i, 2] = np.max(p)
        bars[i, 3] = np.min(p)
        bars[i, 4] = p[-1]
        bars[i, 5] = np.sum(v)
        bars[i, 6] = end - start
        bars[i, 7] = (t[-1] - t[0]) / 60_000_000_000  # nanoseconds to minutes
        bars[i, 8] = t[high_idx]
        bars[i, 9] = t[low_idx]

        indices[i, 0] = start
        indices[i, 1] = end

    return bars, indices


def ticks_to_tick_bars(df: pd.DataFrame, tick_per_bar: int = 1000, col_price: str = "price", col_volume: str = "volume",
    additional_metrics: List[Tuple[Callable, str, List[str]]] = []) -> pd.DataFrame:
    """
    Convert tick-level data into fixed-size tick bars, with optional additional metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Tick DataFrame indexed by datetime, must include price and volume columns.
    tick_per_bar : int, default=1000
        Number of ticks per bar.
    col_price : str, default="price"
        Name of the column containing tick prices.
    col_volume : str, default="volume"
        Name of the column containing tick volumes.
    additional_metrics : List of tuples (function, source, col_names)
        Each tuple consists of:
        - function : callable applied to price, volume, or both slices
        - source   : one of "price", "volume", or "price_volume"
        - col_names: list of column names corresponding to the outputs

    Returns
    -------
    pd.DataFrame
        Tick bars indexed by bar start time, with OHLCV, metadata, and custom metric columns.
    """

    # Convert to NumPy
    prices = df[col_price].to_numpy(np.float64)
    volumes = df[col_volume].to_numpy(np.float64)
    timestamps_ns = df.index.values.astype("int64")

    # Compute bars
    bars_np, index_pairs = _build_tick_bars(prices, volumes, timestamps_ns, tick_per_bar)

    # Base output
    data = {
        "open": bars_np[:, 1],
        "high": bars_np[:, 2],
        "low": bars_np[:, 3],
        "close": bars_np[:, 4],
        "volume": bars_np[:, 5],
        "number_ticks": bars_np[:, 6].astype(int),
        "duration_minutes": bars_np[:, 7],
        "high_time": pd.to_datetime(bars_np[:, 8].astype(np.int64)),
        "low_time": pd.to_datetime(bars_np[:, 9].astype(np.int64)),
    }

    # Add additional metrics
    for func, source, col_names in additional_metrics:
        if source == "price":
            inputs = [func(prices[start:end]) for start, end in index_pairs]
        elif source == "volume":
            inputs = [func(volumes[start:end]) for start, end in index_pairs]
        elif source == "price_volume":
            inputs = [func(prices[start:end], volumes[start:end]) for start, end in index_pairs]
        else:
            raise ValueError(f"Invalid source '{source}'. Must be 'price', 'volume', or 'price_volume'.")

        if isinstance(inputs[0], tuple):
            for i, name in enumerate(col_names):
                data[name] = [out[i] for out in inputs]
        else:
            data[col_names[0]] = inputs

    index = pd.to_datetime(bars_np[:, 0].astype(np.int64))
    return pd.DataFrame(data, index=index).rename_axis("time")