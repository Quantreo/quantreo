# **Getting Started**

Feature engineering is a <u>crucial step</u> in **quantitative trading**. The **Quantreo library** provides several optimized functions to extract meaningful insights from raw market data.

To help you understand how to use each function, we provide **detailed explanations** and **Jupyter Notebook examples** in the [tutorials](/../tutorials/Quantreo-for-beginners)  folder.

!!! warning "📦 Installation & Import"
    Be sure you have installed the quantreo library before running any of these codes. To install it, just run ``pip install quantreo`` in your terminal.

    PS: to import the Features Library Package, you just need to run 
    ```python
    import quantreo.features_engineering as fe
    ```


<br>
## **Data Format**

Quantreo is designed to work with **OHLCV (Open, High, Low, Close, Volume) data**, which is the standard format in financial markets.  


!!! success "🔗 OHLCV focus"
    <u>**Not all features require all columns**</u>. So, if you compute a features needing only the high and the low prices, you can put a dataframe containing only these two variables.
     

Let's create really a very simple dataframe to show you the right format.
```python
import pandas as pd

# Creating a sample OHLCV DataFrame
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
## Features Available

| **Category**        | **Function Name**             | **Quick Explanation**                                                             |
|---------------------|-------------------------------|-----------------------------------------------------------------------------------|
| Candle Information  | `candle_information`          | Returns the candle’s color, its fill state, and its range.                        |
| Candle Information  | `compute_spread`              | Computes the difference between the high and the low (spread).                    |
| Candle Information  | `price_distribution`          | Percentage of closes falling between two dynamic bounds within a rolling window.  |
| Market Regime       | `kama_market_regime`          | Detects market phases (bull / bear) using two KAMA curves.                        |
| Math                | `derivatives`                 | Computes speed (1st derivative) and acceleration (2nd derivative).                |
| Math                | `log_pct`                     | Logarithmic return over a rolling window.                                         |
| Math                | `auto_corr`                   | Rolling autocorrelation on a given column.                                        |
| Math                | `hurst`                       | Computes the Hurst exponent over a rolling window.                                |
| Trend               | `sma`                         | Simple moving average.                                                            |
| Trend               | `kama`                        | Kaufman Adaptive Moving Average (noise-adaptive).                                 |
| Trend               | `linear_slope`                | Slope of a linear regression over a rolling window.                               |
| Volatility          | `close_to_close_volatility`   | Volatility based on the standard deviation of log returns.                        |
| Volatility          | `parkinson_volatility`        | Volatility based on high/low prices only.                                         |
| Volatility          | `rogers_satchell_volatility`  | Volatility that accounts for drift and intraday prices.                           |
| Volatility          | `yang_zhang_volatility`       | Gap-robust volatility combining multiple measures.                                |
