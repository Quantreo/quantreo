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
        low_time[valid]
    )

def ticks_to_time_bars(df: pd.DataFrame, resample_factor: str = "60min", col_price: str = "price", col_volume: str = "volume",
    additional_metrics: List[Tuple[Callable, str, List[str]]] = []) -> pd.DataFrame:
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
    times, opens, highs, lows, closes, vols, counts, start_idxs, end_idxs, high_times, low_times = _build_time_bars(
        prices, volumes, timestamps_ns, window_ns)

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
            data = [func(prices[start:end], volumes[start:end]) for start, end in zip(start_idxs, end_idxs)]
        else:
            raise ValueError(f"Invalid source '{source}'. Must be 'price', 'volume', or 'price_volume'.")

        if isinstance(data[0], tuple):
            for i, name in enumerate(col_names):
                out[name] = [row[i] for row in data]
        else:
            out[col_names[0]] = data

    df_out = pd.DataFrame(out, index=pd.to_datetime(times))
    df_out.index.name = "time"
    return df_out


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
            p = prices[start:i+1]
            v = volumes[start:i+1]
            t = timestamps_ns[start:i+1]

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

            cum_volume = 0.0
            start = i + 1

        i += 1

    return bars, indices


def ticks_to_volume_bars(df: pd.DataFrame, volume_per_bar: float = 1_000_000, col_price: str = "price",
    col_volume: str = "volume", additional_metrics: List[Tuple[Callable, str, List[str]]] = []) -> pd.DataFrame:
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
        return pd.DataFrame(columns=[
            "open", "high", "low", "close", "volume", "number_ticks",
            "duration_minutes", "high_time", "low_time"
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

    # Apply additional metrics (flexible: price, volume, or both)
    for func, source, col_names in additional_metrics:
        if source == "price":
            outputs = [func(prices[start:end]) for start, end in index_pairs]
        elif source == "volume":
            outputs = [func(volumes[start:end]) for start, end in index_pairs]
        elif source == "price_volume":
            outputs = [func(prices[start:end], volumes[start:end]) for start, end in index_pairs]
        else:
            raise ValueError(f"Invalid source '{source}'. Must be 'price', 'volume', or 'price_volume'.")

        if isinstance(outputs[0], tuple):
            for i, name in enumerate(col_names):
                data[name] = [out[i] for out in outputs]
        else:
            data[col_names[0]] = outputs

    index = pd.to_datetime(bars_np[:, 0].astype(np.int64))
    return pd.DataFrame(data, index=index).rename_axis("time")


@njit
def _build_tick_imbalance_bars(prices, volumes, timestamps_ns, expected_imbalance):
    bars = []
    indices = []
    rolling = False
    imbalance = 0.0
    nb_ticks = 0
    start = 0

    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            sign = 1
        elif delta < 0:
            sign = -1
        else:
            continue

        if not rolling:
            start = i
            imbalance = 0.0
            nb_ticks = 0
            rolling = True

        imbalance += sign
        nb_ticks += 1

        if abs(imbalance) > expected_imbalance:
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

            rolling = False

    return bars, indices


def ticks_to_tick_imbalance_bars(df: pd.DataFrame, expected_imbalance: int = 100, col_price: str = "price",
    col_volume: str = "volume", additional_metrics: List[Tuple[Callable, str, List[str]]] = []) -> pd.DataFrame:
    """
    Convert tick-level data into tick imbalance bars, optionally enriched with custom metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Tick DataFrame indexed by datetime, must include price and volume columns.
    expected_imbalance : int, default=100
        Cumulative signed tick imbalance threshold that triggers a new bar.
    col_price : str, default="price"
        Column name representing the price of each tick.
    col_volume : str, default="volume"
        Column name representing the volume of each tick.
    additional_metrics : list of tuples (function, source, col_names)
        Each tuple must contain:
        - function : a callable applied to each bar (1D np.ndarray or 2D if source = 'price_volume')
        - source   : "price", "volume", or "price_volume"
        - col_names: list of output column names returned by the function

    Returns
    -------
    pd.DataFrame
        Tick imbalance bars indexed by bar start time, with OHLCV, metadata, and custom metric columns.
    """
    # Extract numpy arrays
    prices = df[col_price].to_numpy(np.float64)
    volumes = df[col_volume].to_numpy(np.float64)
    timestamps_ns = df.index.values.astype("int64")

    # Generate tick imbalance bars and slicing indexes
    raw_bars, index_pairs = _build_tick_imbalance_bars(prices, volumes, timestamps_ns, expected_imbalance)

    if not raw_bars:
        return pd.DataFrame(columns=[
            "open", "high", "low", "close", "volume",
            "number_ticks", "duration_minutes", "high_time", "low_time"
        ] + [name for _, _, names in additional_metrics for name in names])

    bars_np = np.array(raw_bars)

    # Base OHLCV structure
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


@njit
def _build_tick_run_bars(prices, volumes, timestamps_ns, expected_run):
    bars = []
    indices = []
    start = 1
    count_buy = 0
    count_sell = 0

    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]

        if delta > 0:
            count_buy += 1
        elif delta < 0:
            count_sell += 1
        else:
            continue  # Ignore flat ticks

        max_run = max(count_buy, count_sell)

        if max_run >= expected_run:
            p = prices[start:i+1]
            v = volumes[start:i+1]
            t = timestamps_ns[start:i+1]

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

            start = i + 1
            count_buy = 0
            count_sell = 0

    return bars, indices


def ticks_to_tick_run_bars(df: pd.DataFrame, expected_run: int = 50, col_price: str = "price", col_volume: str = "volume",
    additional_metrics: List[Tuple[Callable[[np.ndarray], float], str, List[str]]] = []) -> pd.DataFrame:
    """
    BETA --> WILL BE MODIFIED IN THE NEXT VERSION OF QUANTREO

    Convert tick-level data into Tick Run Bars (TRBs) based on the dominance of buy or sell activity,
    with support for additional custom metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with tick data indexed by datetime, including price and volume columns.
    expected_run : int, default=50
        Number of buy or sell ticks needed to form a new bar.
    col_price : str, default="price"
        Column name representing the price of each tick.
    col_volume : str, default="volume"
        Column name representing the volume of each tick.
    additional_metrics : list of (function, source, col_names)
        - function : a callable that takes a NumPy array and returns a float or tuple of floats
        - source   : "price" or "volume", defines the array passed to the function
        - col_names: list of column names for the returned values

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by bar start time, with columns:
        ["open", "high", "low", "close", "volume", "number_ticks",
         "duration_minutes", "high_time", "low_time", ...additional metrics]
    """
    prices = df[col_price].to_numpy(np.float64)
    volumes = df[col_volume].to_numpy(np.float64)
    timestamps_ns = df.index.values.astype("int64")

    raw_bars, index_pairs = _build_tick_run_bars(prices, volumes, timestamps_ns, expected_run)

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


@njit
def _build_volume_run_bars(prices, volumes, timestamps, expected_run):
    bars = []
    indices = []

    start = 1  # we start at 1 to compute diffs
    buy_volume = 0.0
    sell_volume = 0.0

    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        sign = 0
        if delta > 0:
            sign = 1
        elif delta < 0:
            sign = -1
        else:
            continue  # skip flat ticks

        if sign == 1:
            buy_volume += volumes[i]
        elif sign == -1:
            sell_volume += volumes[i]

        max_side_volume = max(buy_volume, sell_volume)

        if max_side_volume >= expected_run:
            p = prices[start:i+1]
            v = volumes[start:i+1]
            t = timestamps[start:i+1]

            high_idx = np.argmax(p)
            low_idx = np.argmin(p)

            bars.append((
                t[0],
                p[0],
                np.max(p),
                np.min(p),
                p[-1],
                np.sum(v),
                i + 1 - start,
                (t[-1] - t[0]) / 60_000_000_000,  # nanoseconds to minutes
                t[high_idx],
                t[low_idx]
            ))
            indices.append((start, i + 1))

            start = i + 1
            buy_volume = 0.0
            sell_volume = 0.0

    return bars, indices


def ticks_to_volume_run_bars(df: pd.DataFrame, expected_volume_run: float = 1_000_000, col_price: str = "price",
    col_volume: str = "volume", additional_metrics: List[Tuple[Callable[[np.ndarray], float], str, List[str]]] = []
) -> pd.DataFrame:
    """
    BETA --> WILL BE MODIFIED IN THE NEXT VERSION OF QUANTREO

    Convert tick-level data into volume run bars with optional custom metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with tick data. Must have a DatetimeIndex and price/volume columns.
    expected_volume_run : float
        Volume threshold for one side (buy/sell) to trigger a new bar.
    col_price : str, default="price"
        Name of the column containing tick prices.
    col_volume : str, default="volume"
        Name of the column containing tick volumes.
    additional_metrics : list of (function, source, [col_names])
        List of custom functions applied to either the 'price' or 'volume' slices of each bar.
        Each function must return either a float or a tuple of floats. You must also provide
        the name(s) of the output column(s).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by bar start time with OHLCV, metadata, and optional custom metrics.
    """
    prices = df[col_price].to_numpy(np.float64)
    volumes = df[col_volume].to_numpy(np.float64)
    timestamps = df.index.values.astype("int64")

    raw_bars, index_pairs = _build_volume_run_bars(prices, volumes, timestamps, expected_volume_run)

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
        "low_time": pd.to_datetime(bars_np[:, 9].astype(np.int64)),
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