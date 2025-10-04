# **Getting Started**

In quantitative trading, **target engineering** is as critical as feature engineering. It defines the **objective** your models will learn to predict. The **Quantreo library** provides multiple methods to create custom targets based on **returns**, **events**, and **price magnitudes**.

We also provide **detailed documentation** and **ready-to-use notebooks** inside the [tutorials](/../tutorials/Quantreo-for-beginners) folder to help you master target creation.

??? warning "📦 Installation & Import"
    Be sure you have installed the quantreo library before using these functions. To install it, just run:
    ```bash
    pip install quantreo
    ```
    To import the Target Engineering Package, simply run:
    ```python
    import quantreo.target_engineering as te
    ```

<br>

## **Input Data Format**

Target engineering with Quantreo is primarily based on **OHLCV (Open, High, Low, Close, Volume) data** or **custom price series** depending on the method.

!!! info "Flexible Data Usage"
    Some targets require only the `close` price as input.  
    Additional elements like **volatility** or **peaks & valleys** are automatically computed internally when needed, no extra inputs are required from the user.


Here’s a minimal example to illustrate a basic dataframe you can pass to target functions:

```python
import pandas as pd

# Creating a sample dataframe
data = {
    "open": [100, 102, 101, 103, 105],
    "high": [105, 107, 106, 108, 110],
    "low": [98, 100, 99, 101, 102],
    "close": [102, 104, 103, 105, 107],
    "volume": [10000, 10050, 9950, 10000, 11500]
}
df = pd.DataFrame(data)
print(df)
```


| Open | High | Low | Close | Volume |
|------|------|-----|-------|--------|
| 100  | 105  |  98 |  102  | 10000  |
| 102  | 107  | 100 |  104  | 10050  |
| 101  | 106  |  99 |  103  | 9950   |
| 103  | 108  | 101 |  105  | 10000  |
| 105  | 110  | 102 |  107  | 11500  |

<br>

---
## Targets Available
| **Category** | **Function Name**        | **Quick Explanation**                                                                                                                    |
|-------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Directional | `future_returns_sign`    | Binary target indicating whether the future return is positive.                                                                          |
| Directional | `quantile_label`         | Creates a multi-class target based on quantiles (e.g., bearish, neutral, bullish).                                                       |
| Event-Based | `detect_peaks_valleys`   | Detects structural peaks and valleys in a price series.                                                                                  |
| Magnitude   | `future_returns`         | Future return (log or simple), often used in regression models.                                                                          |
| Magnitude   | `future_volatility`      | Future volatility derived from several methods, such as close-to-close or Parkinson.                                                     |
| Magnitude  | `continuous_barrier_labeling`  | Computes the time (in hours) to reach TP or SL. Inspired by Marco Lopez de Prado's labeling methodology.                                 |
| Directional | `double_barrier_labeling`      | Discrete version of the barrier labeling: returns `1`, `-1`, or `0`.                                                                     |
| Directional | `triple_barrier_labeling`      | Discrete version of the barrier labeling includeing a time limit for labeling. If no barrier is hit within a set duration, label is `0`. |

