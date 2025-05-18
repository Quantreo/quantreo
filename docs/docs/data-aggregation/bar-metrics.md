# **Bar Metrics**

In traditional bar construction, we usually extract standard OHLCV values.  
But sometimes, you want to go further and compute **non-conventional metrics** that provide deeper insight into market microstructure.

This is where **bar-level metrics** come into play.

These functions are designed to work with **tick-level data inside each bar**, allowing you to compute:

- Statistical properties like **skewness**, **kurtosis**, or **Hurst exponent**
- Microstructural features like **volume profile peaks**
- Custom indicators derived from price or volume distributions

Use them with the `additional_metrics` parameter inside any bar-building function to **enrich your bars** with actionable, data-driven signals.

You can find a series of examples on how to create bar aggregations from tick data in the [educational notebooks](/../tutorials/data-aggregation-bar-metrics) provided by Quantreo.

!!! tip "🛠️ How to use additional_metrics"
    The `additional_metrics` parameter lets you enrich any bar with custom columns.

    It must be a **list of tuples**, where each tuple follows this exact structure:

    ``` python
    (
        function,               # A callable applied to the bar's internal data
        "price" | "volume" | "price_volume",  # Data source passed to the function
        ["col_name1", "col_name2", ...]       # Names of the output columns
    )
    ```

    - **`function`**: A Python function that takes a NumPy array (or tuple of arrays if `price_volume`) and returns a float or a tuple of floats.
    - **`"price"` / `"volume"` / `"price_volume"`**: Specifies which data is passed to the function.
    - **`["output_col_name"]`**: The names of the columns added to the resulting DataFrame.

    ✅ These metrics are computed **for each individual bar**, using only the ticks inside that bar.

---
## Available functions
The functions below are designed to be used with the `additional_metrics` parameter in any bar-building function.  
They extract **statistical** and **microstructural** insights from the raw ticks inside each bar.

| **Function**               | **Input**        | **Outputs**                      | **Description**                                                                 | **Parameters**                          |
|----------------------------|------------------|----------------------------------|----------------------------------------------------------------------------------|------------------------------------------|
| `skewness`                 | `"price"`        | `skew`                           | Measures the **asymmetry** of the distribution (right- vs left-skewed).         | —                                        |
| `kurtosis`                 | `"price"`        | `kurt`                           | Measures the **tailedness** (extreme values) of the distribution.               | —                                        |
| `volume_profile_features` | `"price_volume"` | `poc_price`, `poc_position`      | Computes **Volume Profile** over `n_bins` and extracts Point of Control (POC).  | `n_bins` (default=20): number of bins → higher = more precision |
| `max_traded_volume`        | `"price_volume"` | `max_vol`, `price_max_vol`       | Returns the **max tick volume** and the **price** at which it occurred.         | —                                        |

📢 *For a practical example, check out this [educational notebook](/../tutorials/data-aggregation-bar-metrics/#apply-additional-metrics).*

---
## **Custom Metrics**

**Sometimes, standard OHLCV data isn’t enough.**

You might want to extract **advanced metrics** from the raw ticks inside each bar like skewness, volume profile peaks, or volatility spikes.

That’s exactly what `additional_metrics` is for. It lets you plug in your own logic and enrich every bar with custom, computed features.

📢 *For a practical example, check out this [educational notebook](/../tutorials/data-aggregation-bar-metrics/#create-new-metrics).*

<br>

### **Custom Metrics Inputs**
When creating a custom bar metric, your function will receive as input either:

- **A single Numpy array**: `"price"` (the price array for the bar), `"volume"`(the volume array for the bar)
``` python
def your_function(x, **kwargs):
```

- **Two Numpy arrays**: `"price_volume"` (both price and volume arrays `prices, volumes`)
``` python
def your_function(x, y, **kwargs):
```

This choice is defined in the second element of your tuple in `additional_metrics`.  Your function must match the expected input format.

<br>

### **Custom Metrics Logic**

Inside your function, you're free to implement any logic you want.  
You can compute statistical measures (e.g. standard deviation, skewness), custom indicators, or microstructure patterns.

There are **no structural constraints** on what happens inside — as long as the function matches the expected input format (`price`, `volume`, or `price_volume`).

!!! tip "⚡️ Performance Note"
    We highly recommend decorating your function with `@njit` from `numba`.  
    Since bar-building processes involve **millions of ticks**, a pure Python or Pandas implementation can be  **20× to 100× slower** than a Numba-compiled one.

```python
from numba import njit
import numpy as np

@njit
def your_metric(x: np.ndarray) -> float:
    # Compute whatever you want here
    return your_metric_float
```

<br>

### **Custom Metrics Output**

Your function must return either:

- A **single float**  
  → You must provide **one column name** in the `additional_metrics` tuple.

- A **tuple of floats**  
  → You must provide a **list of column names**, one for each value returned.

!!! warning "Important"
    The number of values returned by your function **must match** the number of column names you provide  
    in the `additional_metrics` tuple.  

    Otherwise, you'll get a **runtime error** during bar construction.


✅ The output values will be automatically inserted as new columns in the resulting bar DataFrame.


```python 
# Function that returns one value → one output column
(skewness, "price", ["skew"])

# Function that returns two values → two output columns
(volume_profile_features, "price_volume", ["poc_price", "poc_position"])
```
