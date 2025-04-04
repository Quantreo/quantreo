# **Trend**
You can find a series of examples on how to create these features in the [educational notebooks](/../tutorials/features-engineering-trend) provided by Quantreo.


``` py
import quantreo.features_engineering as fe
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

---

## **Linear Slope**

The `linear_slope` function computes the **slope of a linear regression line** over a rolling window on a given column.  
This slope is often used to detect **momentum shifts**, **trend strength**, or simply **directional bias** over a moving window.

It works by fitting a **simple linear regression** of the form $y = ax + b$ on the rolling window and returning the value of the slope $a$ at each point in time.

A positive slope implies **upward momentum**, a negative slope implies **downward momentum**.



```python title="How to call linear_slope"
fe.trend.linear_slope(df: pd.DataFrame, col: str = 'close', window_size: int = 60)
```

```python title="linear_slope docstring"
"""
Compute the slope of a linear regression line over a rolling window.

This function applies a linear regression on a rolling window of a selected column,
returning the slope of the fitted line at each time step. It uses a fast internal implementation
(`_get_linear_regression_slope`) for efficient computation.

Parameters
----------
df : pandas.DataFrame
    Input DataFrame containing the time series data.
col : str
    Name of the column on which to compute the slope.
window_size : int, optional
    Size of the rolling window used to fit the linear regression (default is 60).

Returns
-------
slope_series : pandas.Series
    A Series containing the slope of the regression line at each time step.
    The first (window_size - 1) values will be NaN due to insufficient data for the initial windows.

Notes
-----
This indicator is useful to assess short- or medium-term price trends.
A positive slope indicates an upward trend, while a negative slope reflects a downward trend.
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-trend/#linear-slope)."
