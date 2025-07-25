# **Bar Building**
You can find a series of examples on how to create bar aggregations from tick data in the [educational notebooks](/../tutorials/data-aggregation-bar-building) provided by Quantreo.

```py
import quantreo.data_aggregation as da
```
<br>

**Input Requirements**

To use these functions, your input `DataFrame` must:

- Be indexed by a **`DatetimeIndex`** (typically named `"datetime"`).
- Contain at least two columns:
  - **`price`** – the transaction price of each tick.
  - **`volume`** – the size of the trade (*can be set to `0` if unknown, but must be present*).

---
## **Ticks to Time Bars**

The `ticks_to_time_bars` function aggregates raw tick data into **fixed-time bars** (e.g., 1-second, 1-minute, etc.). This is the most common form of bar construction, used in nearly all trading platforms.

The function will group ticks by time intervals and compute **OHLCV** values per bar.

It is also possible to add **custom metrics** to each bar using the `additional_metrics` parameter — see the [dedicated tutorial](/../data-aggregation/bar-metrics/#custom-metrics) for a detailed walkthrough.


``` python title="How to call the ticks_to_time_bars function"
def ticks_to_time_bars(df: pd.DataFrame, col_price: str = "price", col_volume: str = "volume",resample_factor: str = "60min",
    additional_metrics: List[Tuple[Callable, str, List[str]]] = []) -> pd.DataFrame:
```

```title="ticks_to_time_bars function docstring"
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
```
📢 *For a practical example, check out this [educational notebook](/../tutorials/data-aggregation-bar-building/#time-bars).*

---

## **Ticks to Tick Bars**

The `ticks_to_tick_bars` function aggregates raw tick data into **fixed-size tick bars** — where each bar contains exactly *N* ticks (e.g., 1,000 ticks per bar). This method preserves microstructure details by standardizing the number of observations per bar rather than the time interval.

The function will sequentially split ticks into equal-sized chunks and compute **OHLCV** values, tick count, duration, and extrema timestamps for each bar.

It is also possible to add **custom metrics** to each bar using the `additional_metrics` parameter — see the [dedicated tutorial](/../data-aggregation/bar-metrics/#custom-metrics) for a detailed walkthrough.

This type of bars comes from the book "Advances in Financial Machine Learning" (Marco Lopez de Prado)

```python title="How to call the ticks_to_tick_bars function"
def ticks_to_tick_bars(df: pd.DataFrame, tick_per_bar: int = 1000, col_price: str = "price", col_volume: str = "volume",
    additional_metrics: List[Tuple[Callable, str, List[str]]] = []) -> pd.DataFrame:
```

```title="ticks_to_tick_bars function docstring"
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
```
📢 *For a practical example, check out this [educational notebook](/../tutorials/data-aggregation-bar-building/#tick-bars).*

---
## **Ticks to Volume Bars**

The `ticks_to_volume_bars` function aggregates raw tick data into bars based on **cumulative traded volume**. A new bar is created every time the specified `volume_per_bar` threshold is reached. This method adapts to market activity: more bars during high activity, fewer during quiet periods.

The function sequentially accumulates volume and computes **OHLCV**, tick count, duration, and extrema timestamps for each volume bar.

It is also possible to add **custom metrics** to each bar using the `additional_metrics` parameter — see the [dedicated tutorial](/../data-aggregation/bar-metrics/#custom-metrics) for a detailed walkthrough.

This type of bars comes from the book "Advances in Financial Machine Learning" (Marco Lopez de Prado)

```python title="How to call the ticks_to_volume_bars function"
def ticks_to_volume_bars(df: pd.DataFrame, volume_per_bar: float = 1_000_000, col_price: str = "price",
    col_volume: str = "volume", additional_metrics: List[Tuple[Callable, str, List[str]]] = []) -> pd.DataFrame:
```

```title="ticks_to_volume_bars function docstring"
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
```
📢 *For a practical example, check out this [educational notebook](/../tutorials/data-aggregation-bar-building/#volume-bars).*


---
## **Ticks to Tick Imbalance Bars**

The `ticks_to_tick_imbalance_bars` function builds bars based on the **signed tick imbalance**. Unlike time- or volume-based bars, a new bar is triggered only when the **cumulative imbalance between buyer-initiated and seller-initiated ticks** exceeds a predefined threshold.

This technique helps normalize market activity by emphasizing **price pressure** rather than time or volume, which is particularly useful for **event-driven strategies** or volatile markets.

The function computes **OHLCV**, tick count, duration, and extrema timestamps for each bar.

It is also possible to add **custom metrics** to each bar using the `additional_metrics` parameter — see the [dedicated tutorial](/../data-aggregation/bar-metrics/#custom-metrics) for a detailed walkthrough.

<br>

**📐 How It Works**

Each incoming tick contributes to a running total based on its **signed direction**:

\[
\text{Signed Tick} = 
\begin{cases}
+1 & \text{if } P_t > P_{t-1} \\
-1 & \text{if } P_t < P_{t-1} \\
0  & \text{otherwise}
\end{cases}
\]

A new bar is created **when the absolute value of the cumulative signed imbalance** exceeds the `expected_imbalance` threshold.

This type of bars comes from the book "Advances in Financial Machine Learning" (Marco Lopez de Prado)


```python title="How to call the ticks_to_tick_imbalance_bars function"
def ticks_to_tick_imbalance_bars(df: pd.DataFrame, expected_imbalance: int = 100, col_price: str = "price",
    col_volume: str = "volume", additional_metrics: List[Tuple[Callable, str, List[str]]] = []) -> pd.DataFrame:
```

```title="ticks_to_tick_imbalance_bars function docstring"
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
```
📢 *For a practical example, check out this [educational notebook](/../tutorials/data-aggregation-bar-building/#tick-imbalance-bars).*

---

## **Ticks to Volume Imbalance Bars**

The `ticks_to_volume_imbalance_bars` function creates bars based on **signed volume imbalance**. A new bar is triggered when the imbalance between buying and selling volume exceeds a predefined threshold.

This method captures **asymmetry in trading pressure**, allowing you to detect key moments where market participation is strongly biased in one direction — often before large price moves.

The function computes **OHLCV**, tick count, duration, extrema timestamps, and optionally, **custom metrics**.

It is also possible to add **custom metrics** to each bar using the `additional_metrics` parameter — see the [dedicated tutorial](/../data-aggregation/bar-metrics/#custom-metrics) for a detailed walkthrough.

<br>

**📐 How It Works**

The volume imbalance is computed as:

\[
\text{Signed Volume}_t =
\begin{cases}
+V_t & \text{if } P_t > P_{t-1} \\
-V_t & \text{if } P_t < P_{t-1} \\
0    & \text{otherwise}
\end{cases}
\]

Where \( V_t \) is the tick volume at time \( t \).  
A new bar is created when the **cumulative sum** of signed volume exceeds `expected_imbalance`.

This type of bars comes from the book "Advances in Financial Machine Learning" (Marco Lopez de Prado)


```python title="How to call the ticks_to_volume_imbalance_bars function"
def ticks_to_volume_imbalance_bars(df: pd.DataFrame, expected_imbalance: float = 500_000, col_price: str = "price",
    col_volume: str = "volume", additional_metrics: List[Tuple[Callable[[np.ndarray], float], str, List[str]]] = []) -> pd.DataFrame:
```

```title="ticks_to_volume_imbalance_bars function docstring"
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
```
📢 *For a practical example, check out this [educational notebook](/../tutorials/data-aggregation-bar-building/#volume-imbalance-bars).*