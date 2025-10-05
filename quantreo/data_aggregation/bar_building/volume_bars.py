import pandas as pd
import numpy as np
from numba import njit
from typing import Callable, List, Tuple


@njit
def _build_volume_bars(prices, volumes, timestamps_ns, volume_per_bar):
    bars = []
    indices = []

    cum_volume = 0.0
    start = 0
    i = 0
    n = len(prices)

    while i < n:
        cum_volume += volumes[i]

        if cum_volume >= volume_per_bar:
            p = prices[start : i + 1]
            v = volumes[start : i + 1]
            t = timestamps_ns[start : i + 1]

            high_idx = np.argmax(p)
            low_idx = np.argmin(p)

            bar = (
                t[0],
                p[0],
                np.max(p),
                np.min(p),
                p[-1],
                np.sum(v),
                i + 1 - start,
                (t[-1] - t[0]) / 60_000_000_000,
                t[high_idx],
                t[low_idx],
            )
            bars.append(bar)
            indices.append((start, i + 1))

            cum_volume = 0.0
            start = i + 1

        i += 1

    return bars, indices


def ticks_to_volume_bars(
    df: pd.DataFrame,
    volume_per_bar: float = 1_000_000,
    col_price: str = "price",
    col_volume: str = "volume",
    additional_metrics: List[Tuple[Callable, str, List[str]]] = [],
) -> pd.DataFrame:
    """
    Convert tick-level data into volume-based bars, optionally enriched with custom metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Tick DataFrame indexed by datetime, must include price and volume columns.
    volume_per_bar : float, default=1_000_000
        Volume threshold that triggers a new bar.
    col_price : str, default="price"
        Column name representing the price of each tick.
    col_volume : str, default="volume"
        Column name representing the volume of each tick.
    additional_metrics : list of tuples (function, source, col_names)
        Each tuple must contain:
        - function : a callable applied to bar slices (can return float or tuple of floats)
        - source   : "price", "volume", or "price_volume"
        - col_names: list of strings (column names returned by the function)

    Returns
    -------
    pd.DataFrame
        Volume bars indexed by bar start time with OHLCV, metadata, and custom metric columns.
    """

    prices = df[col_price].to_numpy(np.float64)
    volumes = df[col_volume].to_numpy(np.float64)
    timestamps_ns = df.index.values.astype("int64")

    # Core bar extraction
    raw_bars, index_pairs = _build_volume_bars(prices, volumes, timestamps_ns, volume_per_bar)

    if not raw_bars:
        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "number_ticks",
                "duration_minutes",
                "high_time",
                "low_time",
            ]
            + [name for _, _, names in additional_metrics for name in names]
        )

    bars_np = np.array(raw_bars)

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

    # Apply additional metrics (flexible: price, volume, or both)
    for func, source, col_names in additional_metrics:
        if source == "price":
            outputs = [func(prices[start:end]) for start, end in index_pairs]
        elif source == "volume":
            outputs = [func(volumes[start:end]) for start, end in index_pairs]
        elif source == "price_volume":
            outputs = [func(prices[start:end], volumes[start:end]) for start, end in index_pairs]
        else:
            raise ValueError(
                f"Invalid source '{source}'. Must be 'price', 'volume', or 'price_volume'."
            )

        if isinstance(outputs[0], tuple):
            for i, name in enumerate(col_names):
                data[name] = [out[i] for out in outputs]
        else:
            data[col_names[0]] = outputs

    index = pd.to_datetime(bars_np[:, 0].astype(np.int64))
    return pd.DataFrame(data, index=index).rename_axis("time")
