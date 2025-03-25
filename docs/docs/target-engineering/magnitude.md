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

