# **Candle Information**
You can find a series of examples on how to create these features in the [educational notebooks](/../tutorials/features-engineering-candle) provided by Quantreo.


``` py
import quantreo.features_engineering import fe
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


