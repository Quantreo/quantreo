# **Magnitude Targets**
You can find several examples on how to create these targets in the [educational notebooks](/../tutorials/target-engineering-magnitude) provided by Quantreo.

```python
import quantreo.target_engineering as te
```

---

## **Future Returns**

The `future_returns` function computes the **amplitude of the future return** for each observation.

- You can choose between **log-returns** or **simple returns**.
- It’s particularly useful for **regression models** or for further feature engineering.

!!! tip "Tip"
    This function is commonly used as a regression target or as input for other quantization methods. <br>
    For example, you can use it on a `picks_and_valleys` target to focus on some specific situations. 

```python title="How to call future_returns"
te.magnitude.future_returns(df: pd.DataFrame, close_col: str = 'close', window_size: int = 10, log_return: bool = True)
```

``` title="future_returns docstring"
"""
Compute future returns over a specified window size.

Parameters
----------
df : pandas.DataFrame
    DataFrame containing price data.
close_col : str, optional
    Name of the column to use as the close price (default is 'close').
window_size : int
    Number of periods to shift forward to calculate the future return.
log_return : bool, optional
    If True, computes the log-return. If False, computes the simple return.

Returns
-------
pandas.Series
    A Series containing the future returns (log or simple).
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/target-engineering-magnitude/#future-returns)."

---

## **Future Volatility**

The `future_volatility` function estimates **future market volatility** using several common models: 
[`close_to_close`](/../features-engineering/volatility/#ctc-volatility), 
[`parkinson`](/../features-engineering/volatility/#parkinson-volatility),
[`rogers_satchell`](/../features-engineering/volatility/#rogers-satchell-volatility),
[`yang_zhang`](/../features-engineering/volatility/#yang-zhang-volatility).


```python title="How to call future_volatility"
te.magnitude.future_volatility(df: pd.DataFrame, method: str = 'close_to_close', window_size: int = 20)
```

``` title="future_volatility docstring"
"""
Compute the volatility over the next 'window_size' periods.

Parameters
----------
df : pandas.DataFrame
    DataFrame containing OHLC or close price data.
method : str
    Volatility method: ['close_to_close', 'parkinson', 'rogers_satchell', 'yang_zhang'].
window_size : int
    Number of periods ahead to estimate volatility.
shift_forward : bool
    If True, shifts volatility back to align with current timestamp.

Returns
-------
pandas.Series
    Future volatility values aligned on the current timestamp.
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/target-engineering-magnitude/#future-volatility)."


---
## **Continuous Barrier Labeling**

The `continuous_barrier_labeling` function estimates the **exact time (in hours)** it takes for the price to hit either a **Take Profit (TP)** or a **Stop Loss (SL)** level, starting from each index.  
It belongs to the family of **event-based targets** and provides **continuous labels** that reflect the **speed** of a movement rather than just its direction.

This labeling approach is useful for **timing analysis**, **position sizing**, and training models that incorporate the notion of **time-to-event**.

!!! warning "⏱ Time-based requirement"
    Unlike other targets, this method **requires**:

    - A **DatetimeIndex** (named `'time'`).
    - Two timestamp columns: `high_time` and `low_time`, indicating **when the high and low of the candle occurred** (not the end of the bar).
    
    These columns are essential to compute the more accurate label possible without using the ticks.

```python title="How to call continuous_barrier_labeling"
te.magnitude.continuous_barrier_labeling(df: pd.DataFrame, open_col: str = "open", high_col: str = "high", low_col: str = "low", high_time_col: str = "high_time",
    low_time_col: str = "low_time", tp: float = 0.015, sl: float = -0.015, buy: bool = True)
```

``` title="continuous_barrier_labeling docstring"
"""
Compute the time (in hours) to hit either a Take Profit (TP) or Stop Loss (SL) level
after entering a trade, using a fast Numba-accelerated barrier labeling method.

Parameters
----------
df : pandas.DataFrame
    Input DataFrame with a DatetimeIndex named 'time'.
    Must include the following columns:
    - Price: open_col, high_col, low_col
    - Timestamps: high_time_col, low_time_col (datetime when the high/low occurred)
open_col : str, optional
    Column name for the open price (default is 'open').
high_col : str, optional
    Column name for the high price (default is 'high').
low_col : str, optional
    Column name for the low price (default is 'low').
high_time_col : str, optional
    Column name for the timestamp when the high occurred (default is 'high_time').
low_time_col : str, optional
    Column name for the timestamp when the low occurred (default is 'low_time').
tp : float, optional
    Take Profit threshold (as % variation from open). Must be > 0.
sl : float, optional
    Stop Loss threshold (as % variation from open). Must be < 0.
buy : bool, optional
    Whether to simulate a long trade (True) or a short trade (False). Default is True.

Returns
-------
pandas.Series
    A Series with the same index as the input DataFrame.
    Each value represents the time in **hours** before hitting TP or SL:
    - Positive value → TP was hit first.
    - Negative value → SL was hit first.
    - Zero → Neither was hit or data ended too early.
"""
```
📢 "For a practical example, check out the [educational notebook](/../tutorials/target-engineering-magnitude/#continuous-barrier-labeling)."
