# **Candle Information**
You can find a series of examples on how to create these features in the [educational notebooks](/../tutorials/features-engineering-candle) provided by Quantreo.


``` py
import quantreo.features_engineering as fe
```

---

## **Basic Candle Information**

The `candle_information` function provides essential insights about a candle by calculating the following features:

- **The Candle Way**: This feature returns `-1` if the candle is red (indicating a negative variation from the open price to the close price) or `1` if the candle is green (indicating a positive variation from the open price to the close price).

- **The Filling**: This feature computes the ratio between the absolute difference from the open to the close price and the total range of the candle (i.e., the difference between the high and low prices). $\frac{| \text{close} - \text{open} |}{| \text{high} - \text{low} |}$

- **The Amplitude**: This feature measures the relative price movement of a candle compared to its average. $\frac{| \text{close} - \text{open} |}{\left( \frac{\text{open} + \text{close}}{2} \right)}$


```python title="How to call the candle_information function"
fe.candle.candle_information(df: pd.DataFrame, open_col: str = 'open', high_col: str = 'high',
                       low_col: str = 'low', close_col: str = 'close')
```
``` title="candle_information function docstring"
"""
Compute candle information indicators for a given OHLC DataFrame.

This function calculates:
  - 'candle_way': Indicator for the candle's color (1 if close > open, -1 otherwise).
  - 'filling': The filling percentage, computed as the absolute difference between
               close and open divided by the range (high - low).
  - 'amplitude': The candle amplitude as a percentage, calculated as the absolute difference
                 between close and open divided by the average of open and close, multiplied by 100.

Parameters
----------
df : pandas.DataFrame
    DataFrame containing OHLC data.
open_col : str, optional
    Column name for open prices (default is 'open').
high_col : str, optional
    Column name for high prices (default is 'high').
low_col : str, optional
    Column name for low prices (default is 'low').
close_col : str, optional
    Column name for close prices (default is 'close').

Returns
-------
Tuple[pd.Series, pd.Series, pd.Series]
    - candle_way (pd.Series[int]): The direction of the candle (`1` for bullish, `-1` for bearish).
    - filling (pd.Series[float]): The proportion of the candle range occupied by the body.
    - amplitude (pd.Series[float]): The relative size of the candle in percentage.
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-candle/#basic-candle-information)."

!!! tip "Tip"
    These three variables (`candle_way`, `filling` and `amplitude`) are often very correlated. Do not hesitate to keep only the best feature related to your problem or combine them smartly into one variable.


---

## **Spread Calculation**

The `compute_spread` function provides a simple yet valuable insight into the **difference between the highest and lowest price** within a given period. This metric is useful for analyzing intra-bar market volatility.

  $$ \text{spread} = \text{high} - \text{low} $$

The higher the `spread` value is, the more the market is volatile.

```python title="How to call the compute_spread function"
fe.candle.compute_spread(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low')
```

``` title="spread function docstring"
"""
Compute the spread between the high and low price columns.

Parameters
----------
df : pandas.DataFrame
    The DataFrame containing price data.
high_col : str, optional
    Column name for the high prices (default is 'high').
low_col : str, optional
    Column name for the low prices (default is 'low').

Returns
-------
spread_series : pandas.Series
    A Series indexed the same as `df`, containing the spread values.
"""
```
📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-candle/#spread)."

---

## **Price Distribution**

The `price_distribution` function calculates the **percentage of closing prices** that lie within a given **relative range** of their **rolling high-low interval**.

This feature is especially helpful to identify **price compression** (when many prices cluster in the center of the range) or **volatility bursts** (when price spreads out to the extremes).

It works by:

- Computing the **min** and **max** price in each rolling window.

- Evaluating the **number of prices** that fall between two dynamic thresholds: `start_percentage` (e.g., 0.25) of the range and `end_percentage` (e.g., 0.75) of the range


```python title="How to call price_distribution"
fe.candle.price_distribution(df: pd.DataFrame, col: str = 'close', window_size: int = 60,
                            start_percentage: float = 0.25, end_percentage: float = 0.75)
```

```python title="price_distribution docstring"
"""
Compute the percentage of close prices within a relative range of their local low-high interval,
over a rolling window.

This function calculates, for each window, how many values lie within a given percentage band
of the [low, high] range. It is useful to detect price compression or expansion around a zone.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the time series data.
col : str
    Name of the column containing the close prices.
window_size : int, optional
    Size of the rolling window (default is 60).
start_percentage : float, optional
    Start of the relative range as a percentage of (high - low). Default is 0.25 (25%).
end_percentage : float, optional
    End of the relative range as a percentage of (high - low). Default is 0.75 (75%).

Returns
-------
pd.Series
    Series with the same index as the input, containing the computed percentage values for each window.
    First (window_size - 1) rows will be NaN.
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-candle/#price-distribution)."
