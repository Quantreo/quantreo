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

!!! tip "Tip"
    The flexibility of `quantile_label()` makes it suitable for **dynamic labeling** in trading systems, e.g., to build multi-class models that adapt to market regimes.

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


