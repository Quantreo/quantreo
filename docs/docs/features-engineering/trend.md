# **Trend**
You can find a series of examples on how to create these features in the [educational notebooks](/../tutorials/features-engineering-trend) provided by Quantreo.


``` py
import quantreo.features_engineering import fe
```

---

## **Simple Moving Average**

The `sma` function computes a Simple Moving Average (SMA) on any numerical column of your DataFrame. The SMA is one of the most widely used technical indicators to smooth out price data and detect trends over a given period.

!!! warning
    The first `(window_size - 1)` values will return `NaN` due to insufficient data to compute the average on these points.


```python title="How to call the sma function"
fe.trend.sma(df: pd.DataFrame, col: str = 'close', window_size: int = 30)
```

``` title="sma function docstring"
"""
Calculate the Simple Moving Average (SMA) using Pandas rolling.mean.

Parameters
----------
df : pandas.DataFrame
    DataFrame containing the input data.
col : str
    Name of the column on which to compute the SMA.
window_size : int, optional
    The window size for computing the SMA (default is 30).

Returns
-------
sma_series : pandas.Series
    A Series indexed the same as the input DataFrame, containing the SMA values.
    The first (window_size - 1) entries will be NaN due to insufficient data.
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-trend/#simple-moving-average)."


---
## **Kaufman's Adaptive Moving Average (KAMA)**

The `kama` function calculates the Kaufman's Adaptive Moving Average, which adjusts dynamically to market noise by adapting its smoothing factor based on the price efficiency ratio.

- **KAMA** belongs to the family of moving averages and was developed by Perry J. Kaufman. It was first introduced in his book *"Smarter Trading: Improving Performance in Changing Markets" (1995)*.
- This indicator is designed to smooth prices when markets are ranging and to be more responsive when markets trend.

The calculation is based on:

$$
ER_t = \frac{|\text{Price}_t - \text{Price}_{t - l_1}|}{\sum_{i = t - l_1 + 1}^{t} |\text{Price}_i - \text{Price}_{i - 1}|}
$$

Then, the smoothing constant:

$$
SC_t = \left( ER_t \cdot \left( \frac{2}{l_2 + 1} - \frac{2}{l_3 + 1} \right) + \frac{2}{l_3 + 1} \right)^2
$$

Finally, the recursive calculation:

$$
KAMA_t = KAMA_{t-1} + SC_t \cdot \left( \text{Price}_t - KAMA_{t-1} \right)
$$

!!! warning
    The first `(l1 - 1)` values will be `NaN` due to insufficient data for the efficiency ratio calculation.


```python title="How to call the kama function"
fe.trend.kama(df: pd.DataFrame, col: str = 'close', l1: int = 10, l2: int = 2, l3: int = 30)
```

``` title="kama function docstring"
"""
Calculate Kaufman's Adaptive Moving Average (KAMA) for a specified column in a DataFrame.

KAMA adapts to market noise by adjusting its smoothing constant based on an efficiency ratio.

Parameters
----------
df : pandas.DataFrame
    DataFrame containing the price data.
col : str
    Column name on which to compute the KAMA.
l1 : int, optional
    Rolling window length for computing the efficiency ratio (default is 10).
l2 : int, optional
    Parameter for the fastest EMA constant (default is 2).
l3 : int, optional
    Parameter for the slowest EMA constant (default is 30).

Returns
-------
pandas.Series
    A Series containing the computed KAMA values, indexed the same as `df` and named "kama".
    The first (l1 - 1) values will likely be NaN due to insufficient data.
"""

```
📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-trend/#kaufmans-adaptive-moving-average-kama)."
