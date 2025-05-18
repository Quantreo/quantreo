# **Getting Started**

Data aggregation is a <u>fundamental step</u> in transforming raw tick data into structured insights. The **Quantreo library** provides tools to create **alternative bars** (like tick, volume, or imbalance bars) and compute **intra-candle metrics**.

These tools help bridge the gap between raw microstructure data and usable features for backtesting or modeling.

To better understand how to use each function, you’ll find **detailed explanations** and **Jupyter Notebook examples** in the [tutorials](/../tutorials/Quantreo-for-beginners) folder.

!!! warning "📦 Installation & Import"
    Make sure the Quantreo library is installed before running any code. You can install it with:

    ```
    pip install quantreo
    ```
    
    Then import the data aggregation tools with:
    ```python
    import quantreo.data_aggregation as da
    ```

---
## Bar Building Input Format

To build bars (time bars, tick bars, volume bars, etc.), Quantreo expects **tick-level data** — that is, individual trades with at least a **price** and a **volume**.

This raw format allows you to recreate flexible and informative bars that preserve market microstructure.

!!! success "🔗 Minimal requirements"
    <u>**Only two columns are required**</u>: one for the price and one for the volume.  
    The index must be a `DatetimeIndex`, as bar-building relies on time-based ordering.

Let’s create a minimal example to demonstrate the expected format:

```python
import pandas as pd

# Sample tick data
data = {
    "price": [100.0, 100.1, 100.2, 100.15, 100.05],
    "volume": [3, 1, 2, 1, 5]
}
index = pd.date_range("2023-01-01 09:30:00", periods=5, freq="S")
df = pd.DataFrame(data, index=index)

print(df)
```

| Datetime             | Price  | Volume |
|----------------------|--------|--------|
| 2023-01-01 09:30:00  | 100.00 | 3      |
| 2023-01-01 09:30:01  | 100.10 | 1      |
| 2023-01-01 09:30:02  | 100.20 | 2      |
| 2023-01-01 09:30:03  | 100.15 | 1      |
| 2023-01-01 09:30:04  | 100.05 | 5      |

---

## Bar Building Functions

| Category         | Function Name                      | Quick Explanation                                                                 |
|------------------|-------------------------------------|------------------------------------------------------------------------------------|
| Time-based       | `ticks_to_time_bars`               | Aggregates ticks into bars based on fixed time intervals (e.g., 1 min, 4H).        |
| Tick-based       | `ticks_to_tick_bars`               | Creates bars with a fixed number of ticks per bar (e.g., 1000 ticks per bar).      |
| Volume-based     | `ticks_to_volume_bars`             | Forms bars when a target cumulative volume threshold is reached.                   |
| Imbalance-based  | `ticks_to_tick_imbalance_bars`     | Creates a bar when the signed tick imbalance exceeds a defined threshold.          |
| Imbalance-based  | `ticks_to_volume_imbalance_bars`   | Creates a bar when the signed volume imbalance exceeds a defined threshold.        |
