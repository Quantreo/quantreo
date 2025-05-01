# **Directional Targets**
You can find several examples on how to create these targets in the [educational notebooks](/../tutorials/target-engineering-directional) provided by Quantreo.

```python
import quantreo.target_engineering as te
```

---

## **Future Returns Sign**

The `future_returns_sign` function generates a **binary directional target** based on future returns.

- It computes the return between the current price and the price `window_size` steps ahead.
- Then it assigns a **positive label** (default `1`) if the return is strictly positive, otherwise a **negative label** (default `0`).

This target is ideal for **trend-following models** or binary classification.

```python title="How to call future_returns_sign"
te.directional.future_returns_sign(df: pd.DataFrame, close_col: str = 'close', window_size: int = 10, log_return: bool = True)
```

``` title="future_returns_sign docstring"
"""
Generate a directional target by computing future returns and binarizing them.

Parameters
----------
df : pandas.DataFrame
    DataFrame containing price data.
close_col : str, optional
    Name of the column to use as the close price (default is 'close').
window_size : int, optional
    Number of periods to shift forward (default is 10).
log_return : bool, optional
    If True, computes log-return, else simple return (default is True).
positive_label : int, optional
    Value for strictly positive returns (default is 1).
negative_label : int, optional
    Value for zero or negative returns (default is 0).

Returns
-------
pandas.Series
    A Series containing binary directional labels.
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/target-engineering-directional/#future-returns-sign)."

---

## **Quantile Label Target**

The `quantile_label` function is designed to generate **quantile-based multi-class labels** for target engineering.

- It assigns a **positive label** to values greater than a computed or provided upper quantile (e.g., the 67th percentile).
- It assigns a **negative label** to values lower than a computed or provided lower quantile (e.g., the 33rd percentile).
- It assigns a **neutral label** to all values that fall between the two quantiles.

Additionally, this function allows you to **predefine thresholds** (`q_high`, `q_low`) or return them along with the labels if `return_thresholds=True`. This is particularly useful for building classification datasets with clearly segmented zones such as **bullish**, **neutral**, and **bearish** phases.

!!! danger "Quantile Target Caution"
    When using quantile-based targets, always compute the **quantile thresholds on the training set only**, and apply them to the test set to generate labels.  
    Computing quantiles on the full dataset would introduce a **look-ahead bias**, leaking future information into your model.


```python title="How to call quantile_label"
te.directional.quantile_label(df: pd.DataFrame, col: str, upper_quantile_level: float = 0.67,
                   lower_quantile_level: float | None = None, q_high: float | None = None, q_low: float | None = None,
                   return_thresholds: bool = False, positive_label: int = 1, negative_label: int = -1,
                   neutral_label: int = 0) -> pd.Series | tuple[pd.Series, float, float]:
```

``` title="quantile_label docstring"
"""
Generate quantile-based labels (positive/neutral/negative) and optionally return computed thresholds.

Parameters
----------
df : pandas.DataFrame
    DataFrame containing the target column.
col : str
    Name of the column used to compute quantiles (e.g., 'fut_ret').
upper_quantile_level : float, optional
    The quantile threshold for the upper label (default is 0.67).
lower_quantile_level : float or None, optional
    The quantile threshold for the lower label (defaults to 1 - upper_quantile_level).
q_high : float or None, optional
    Predefined upper quantile threshold (bypass quantile computation).
q_low : float or None, optional
    Predefined lower quantile threshold (bypass quantile computation).
return_thresholds : bool, optional
    If True, returns both the labels and the computed thresholds.
positive_label : int or any, optional
    Label for values strictly above the upper quantile (default is 1).
negative_label : int or any, optional
    Label for values strictly below the lower quantile (default is -1).
neutral_label : int or any, optional
    Label for values between the quantiles (default is 0).

Returns
-------
pd.Series
    A pandas Series of labeled values.
q_high : float (optional)
    Upper quantile threshold if `return_thresholds=True`.
q_low : float (optional)
    Lower quantile threshold if `return_thresholds=True`.
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/target-engineering-directional/#quantile-label)."


---
## **Double Barrier Labeling**

The `double_barrier_labeling` function transforms the **continuous time-to-barrier** output into a **classification target**.  
It assigns a **discrete label** depending on whether the price hit the **Take Profit (TP)** or **Stop Loss (SL)** level first.

This method is particularly useful for building **binary classifiers** or **directional models** in event-based trading strategies.

!!! danger "⏱ Time-based requirement"
    Like the continuous version, this method **requires**:

    - A **DatetimeIndex** (named `'time'`).
    - Two timestamp columns: `high_time` and `low_time`, indicating **when the high and low occurred**.

    These allow the function to infer the correct sequence of TP/SL hits with **high precision**.

```python title="How to call double_barrier_labeling"
te.directional.double_barrier_labeling(df: pd.DataFrame, open_col: str = "open", high_col: str = "high", low_col: str = "low",
    high_time_col: str = "high_time", low_time_col: str = "low_time", tp: float = 0.015, sl: float = -0.015, buy: bool = True)
```

``` title="double_barrier_labeling docstring"
"""
Compute double barrier classification labels based on TP/SL logic.

This function wraps `continuous_barrier_labeling` and converts the continuous
duration-based output into discrete labels:
    - 1  → Take Profit was hit first
    - -1 → Stop Loss was hit first
    - 0  → No barrier hit within max horizon

Parameters
----------
df : pd.DataFrame
    Input DataFrame with a DatetimeIndex named 'time'.
    Must include the following columns:
    - Price: open_col, high_col, low_col
    - Timestamps: high_time_col, low_time_col (datetime when the high/low occurred)
open_col : str, optional
    Name of the column containing the open price (default is 'open').
high_col : str, optional
    Name of the column containing the high price (default is 'high').
low_col : str, optional
    Name of the column containing the low price (default is 'low').
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
    A Series of labels indicating the event outcome:
    - 1 → TP hit first
    - -1 → SL hit first
    - 0 → No barrier was hit
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/target-engineering-directional/#double-barrier-labeling)."


---
## **Triple Barrier Labeling**

The `triple_barrier_labeling` function generates **discrete classification labels** based on **three criteria**:

- A **Take Profit (TP)** level,
- A **Stop Loss (SL)** level,
- A **maximum holding duration** (in hours).

It extends the **double barrier** approach by assigning a **neutral label (0)** when neither barrier is reached within a given time limit.

This target is especially useful in **backtesting** and **reinforcement learning**, where decisions depend not only on price movement but also on **timing constraints**.

!!! danger "⏱ Time-based requirement"
    This method **requires**:

    - A **DatetimeIndex** named `'time'`.
    - Two timestamp columns: `high_time` and `low_time`, representing the **exact time the high and low were hit**.
    - A value for `max_duration_h` (the **maximum holding time** allowed).

```python title="How to call triple_barrier_labeling"
te.directional.triple_barrier_labeling(df: pd.DataFrame, max_duration_h: float,
    open_col: str = "open", high_col: str = "high", low_col: str = "low",
    high_time_col: str = "high_time", low_time_col: str = "low_time",
    tp: float = 0.015, sl: float = -0.015, buy: bool = True)
```

``` title="triple_barrier_labeling docstring"
"""
Compute triple barrier classification labels based on TP/SL and a max holding time.

This function wraps `continuous_barrier_labeling` and converts its continuous time output into:
    -  1 → TP was hit within the maximum duration
    - -1 → SL was hit within the maximum duration
    -  0 → Neither was hit in time (timeout)

Parameters
----------
df : pd.DataFrame
    Input DataFrame with a DatetimeIndex named 'time'.
    Must include the following columns:
    - Price: open_col, high_col, low_col
    - Timestamps: high_time_col, low_time_col (when the high/low occurred)
max_duration_h : float
    Maximum allowed time (in hours) to hit a barrier.
open_col : str, optional
    Column name for the open price (default is 'open').
high_col : str, optional
    Column name for the high price (default is 'high').
low_col : str, optional
    Column name for the low price (default is 'low').
high_time_col : str, optional
    Timestamp column for the high point (default is 'high_time').
low_time_col : str, optional
    Timestamp column for the low point (default is 'low_time').
tp : float, optional
    Take Profit threshold (must be > 0).
sl : float, optional
    Stop Loss threshold (must be < 0).
buy : bool, optional
    Whether to simulate a long (True) or short (False) trade.

Returns
-------
pandas.Series
    A Series of labels for each row:
    - 1 → TP hit before SL and within max time
    - -1 → SL hit before TP and within max time
    - 0 → Timeout (neither TP nor SL hit within the allowed time)
"""

```
📢 "For a practical example, check out the [educational notebook](/../tutorials/target-engineering-directional/#triple-barrier-labeling)."

