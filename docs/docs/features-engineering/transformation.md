# **Transformation**

The `transformation` module contains functions that **reshape, rescale, or smooth raw price data** to make it more informative or statistically stable before further analysis.

You can find multiple examples of how to apply these transformations in the [educational notebooks](/../tutorials/transformation) provided by Quantreo.


``` py
import quantreo.features_engineering as fe
```

---
## **Fisher Transform**

The `fisher_transform` function applies a mathematical transformation that converts **non-Gaussian price data into a distribution resembling a normal curve**.

This is particularly useful for **highlighting turning points**, because values tend to cluster near the center and **amplify at the extremes** (±2 or more), making overbought and oversold conditions easier to detect.

The formula is:

\[
x_t = \frac{2 \cdot \left( \text{median}_t - \min_t \right)}{\max_t - \min_t} - 1
\quad \Rightarrow \quad
\text{Fisher}_t = 0.5 \cdot \ln\left( \frac{1 + x_t}{1 - x_t} \right)
\]

- `x_t` is a normalized representation of the price at time `t`, scaled between -1 and 1.
- The Fisher Transform **amplifies edge behavior**, making it a good signal enhancer.

```python title="How to call fisher_transform"
fe.transformation.fisher_transform(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low', window_size: int = 10)
```

``` title="fisher_transform function docstring"
"""
Compute the Fisher Transform indicator.

The Fisher Transform maps price data into a Gaussian-like distribution
using the formula:
    Fisher = 0.5 * ln((1 + x) / (1 - x))

Where:
    x = 2 * (median - min) / (max - min) - 1

It is typically used to detect turning points and overbought/oversold conditions.

Args:
    df (pd.DataFrame): DataFrame containing OHLC price data.
    high_col (str): Column name for the high price.
    low_col (str): Column name for the low price.
    window_size (int): Rolling window used to normalize the price range.

Returns:
    pd.Series: A Series containing the Fisher Transform values.
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-transformation/#fisher-transform)."

---

## **Savitzky-Golay Filter (Causal)**

The `savgol_filter` function applies a **causal version of the Savitzky-Golay filter** to a time series.

This transformation fits a **polynomial of order `k`** over a **rolling window** of past values and evaluates the value of that polynomial at the **last point** of the window. It is widely used to **smooth noisy signals** while preserving **local structure** like peaks and inflection points.

!!! warning "Causal vs Standard Savitzky-Golay"

    This implementation differs from the classic `scipy.signal.savgol_filter`, which applies a **centered filter** that includes both past and future values — causing **look-ahead bias** in real-time trading.
    
    Here, the filter is applied in a **rolling**, meaning it only uses **past observations** within the window.  
    This makes it **safe for backtesting and live usage**, while preserving most of the smoothing benefits of the standard version.


```python title="How to call savgol_filter"
fe.transformation.savgol_filter(df: pd.DataFrame,  col: str = 'close', window_size: int = 11, polyorder: int = 3)
```

``` title="savgol_filter function docstring"
"""
Compute a causal Savitzky-Golay filter using optimized matrix operations.

This version reproduces the behavior of:
    df[col].rolling(window_size).apply(lambda x: savgol_filter(x, window_size, polyorder)[-1])

It applies a rolling polynomial regression over a past-only window, avoiding look-ahead bias.

Args:
    df (pd.DataFrame): DataFrame containing the input series.
    col (str): Column name of the series to smooth.
    window_size (int): Length of the rolling window (must be odd).
    polyorder (int): Degree of the fitted polynomial (must be < window_size).

Returns:
    pd.Series: Smoothed series using the causal Savitzky-Golay filter.
"""
```
📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-transformation/#savitzky-golay-filter)."
