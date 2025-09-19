import pandas as pd
import numpy as np
from numba import njit
from typing import Callable, List, Tuple


@njit
def _build_volume_imbalance_bars(prices, volumes, timestamps_ns, expected_imbalance):
    bars = []
    indices = []

    start = 1
    cum_imbalance = 0.0

    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            sign = 1
        elif delta < 0:
            sign = -1
        else:
            continue

        volume_signed = sign * volumes[i]
        cum_imbalance += volume_signed

        if abs(cum_imbalance) >= expected_imbalance:
            p = prices[start:i + 1]
            v = volumes[start:i + 1]
            t = timestamps_ns[start:i + 1]

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
                t[low_idx]
            )
            bars.append(bar)
            indices.append((start, i + 1))

            cum_imbalance = 0.0
            start = i + 1

    return bars, indices


def ticks_to_volume_imbalance_bars(df: pd.DataFrame, expected_imbalance: float = 500_000, col_price: str = "price",
    col_volume: str = "volume", additional_metrics: List[Tuple[Callable[[np.ndarray], float], str, List[str]]] = []) -> pd.DataFrame:
    """
    Convert tick-level data into volume imbalance bars, optionally enriched with custom metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Tick DataFrame indexed by datetime, must include price and volume columns.
    expected_imbalance : float, default=500_000
        Signed volume imbalance threshold that triggers a new bar.
    col_price : str, default="price"
        Column name representing the price of each tick.
    col_volume : str, default="volume"
        Column name representing the volume of each tick.
    additional_metrics : list of tuples (function, source, col_names)
        - function : a callable that takes a NumPy slice (1D array) and returns a float or tuple of floats.
        - source   : "price" or "volume", defines what data is passed to the function.
        - col_names: list of names corresponding to the outputs of the function.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by bar start time, with columns:
        ["open", "high", "low", "close", "volume", "number_ticks",
         "duration_minutes", "high_time", "low_time", ...custom metric columns]
    """
    prices = df[col_price].to_numpy(dtype=np.float64)
    volumes = df[col_volume].to_numpy(dtype=np.float64)
    timestamps_ns = df.index.values.astype("int64")

    raw_bars, index_pairs = _build_volume_imbalance_bars(prices, volumes, timestamps_ns, expected_imbalance)

    if not raw_bars:
        return pd.DataFrame(columns=[
            "open", "high", "low", "close", "volume",
            "number_ticks", "duration_minutes", "high_time", "low_time"
        ] + [name for _, _, names in additional_metrics for name in names])

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
        "low_time": pd.to_datetime(bars_np[:, 9].astype(np.int64))
    }

    # Additional metrics computation
    for func, source, col_names in additional_metrics:
        if source == "price":
            inputs = [prices[start:end] for start, end in index_pairs]
        elif source == "volume":
            inputs = [volumes[start:end] for start, end in index_pairs]
        elif source == "price_volume":
            inputs = [(prices[start:end], volumes[start:end]) for start, end in index_pairs]
        else:
            raise ValueError(f"Invalid source '{source}'. Must be 'price', 'volume', or 'price_volume'.")

        outputs = [func(*x) if isinstance(x, tuple) else func(x) for x in inputs]

        if isinstance(outputs[0], tuple):
            for i, name in enumerate(col_names):
                data[name] = [out[i] for out in outputs]
        else:
            data[col_names[0]] = outputs

    index = pd.to_datetime(bars_np[:, 0].astype(np.int64))
    return pd.DataFrame(data, index=index).rename_axis("time")