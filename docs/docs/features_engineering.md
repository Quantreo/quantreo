# **Features Engineering**

Feature engineering is a <u>crucial step</u> in **quantitative trading**. The **Quantreo library** provides several optimized functions to extract meaningful insights from raw market data.

To help you understand how to use each function, we provide **detailed explanations** and **Jupyter Notebook examples** in the `examples/` folder.

!!! warning "Warning"
    Be sure you have installed the quantreo library before running any of these codes. To install it, just run ``pip install quantreo`` in your terminal.

!!! danger "Danger"
    <u>**PARLER DU FORMAT DES DONNÉES**</u>

---

## **Candle Information**
You can find a series of examples on how to create these features in the [educational notebooks](https://www.quantreo.com) provided by [Quantreo](https://www.quantreo.com).


``` py
from quantreo.features_engineering import candle
```

---

### **Basic Candle Information**

The `candle_information` function provides essential insights about a candle by calculating the following features:

- **The Candle Way**: This feature returns `-1` if the candle is red (indicating a negative variation from the open price to the close price) or `1` if the candle is green (indicating a positive variation from the open price to the close price).

- **The Filling**: This feature computes the ratio between the absolute difference from the open to the close price and the total range of the candle (i.e., the difference between the high and low prices). $\frac{| \text{close} - \text{open} |}{| \text{high} - \text{low} |}$

- **The Amplitude**: This feature measures the relative price movement of a candle compared to its average. $\frac{| \text{close} - \text{open} |}{\left( \frac{\text{open} + \text{close}}{2} \right)}$


```python
def candle_information(df: pd.DataFrame, open_col: str = 'open', high_col: str = 'high',
                       low_col: str = 'low', close_col: str = 'close') 
                       -> Tuple[pd.Series, pd.Series, pd.Series]:
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

📢 "For a practical example, check out the [educational notebook](examples/Features_Engineering_Candle.ipynb)."

!!! tip "Tip"
    These three variables (`candle_way`, `filling` and `amplitude`) are often very correlated. Do not hesitate to keep only the best feature related to your problem or combine them smartly into one variable.


---

### **Spread Calculation**

The `compute_spread` function provides a simple yet valuable insight into the **difference between the highest and lowest price** within a given period. This metric is useful for analyzing intra-bar market volatility.

  $$ \text{spread} = \text{high} - \text{low} $$

The higher the `spread` value is, the more the market is volatile.

```python
def compute_spread(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low') -> pd.Series:
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
📢 "For a practical example, check out the [educational notebook](https://www.quantreo.com)."



---
## **Volatility**

!!! danger "à rajouter"
    N'oubliez pas que la valeur de la volatilité va varier en fonction du timeframe utilisé

``` py
from quantreo.features_engineering import volatility
```



---
### **CTC Volatility**

The `close_to_close_volatility` function calculates the **close-to-close volatility**, a widely used measure of market risk based on the standard deviation of **log returns**. This approach captures **price fluctuations over time** and is particularly useful in **statistical modeling and risk management**. The **close-to-close volatility** is defined as:

\[
\sigma_{CC} = \sqrt{\frac{1}{N - 1} \sum_{i=1}^{N} \left( \ln \frac{C_i}{C_{i-1}} - \mu \right)^2}
\]

Where:

- **\(C_i\)**: The **closing price** at time \(i\).
- **\(N\)**: The number of periods in the rolling window.
- **\(\mu\)**: The mean of the log returns over the window.


```python
def close_to_close_volatility(df: pd.DataFrame, close_col: str = 'close', window_size: int = 30) -> pd.Series:
    """
    Calculate the rolling close-to-close volatility.md using standard deviation.

    This method computes the rolling standard deviation of the log returns, 
    which represents the close-to-close volatility.md over a specified window.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the price data.
    close_col : str, optional
        Column name for the closing prices (default is 'close').
    window_size : int, optional
        The number of periods to include in the rolling calculation (default is 30).

    Returns
    -------
    volatility_series : pd.Series
        A Series indexed the same as `df`, containing the rolling close-to-close volatility.md.
        The first `window_size` rows will be NaN because there is insufficient data
        to compute the volatility.md in those windows.
    """
```


📢 *For a practical example, check out the [educational notebook](https://www.quantreo.com).*

---
### **Parkinson Volatility**

The `parkinson_volatility` function computes the **Parkinson volatility**, an estimator designed to measure market volatility using only **high and low prices**. This approach provides a more accurate estimate than simple **close-to-close volatility**, particularly in **markets with low closing price variation but significant intraday movement**.

!!! tip "Tip"
    Unlike standard volatility measures, the **Parkinson estimator** is a **rolling volatility measure**, meaning that at each time step, the function calculates the volatility over the **last N observations**, producing a **time series of volatility values**.

The **Parkinson estimator** is defined as:

\[
\sigma_P^2 = \frac{1}{4N \ln(2)} \sum_{i=1}^{N} \left( \ln \frac{h_i}{l_i} \right)^2
\]

*(Source: Parkinson, 1980, "The Extreme Value Method for Estimating the Variance of the Rate of Return")*

Where:

- **\(h_i\)**: The **high price** at time \(i\).
- **\(l_i\)**: The **low price** at time \(i\).
- **\(N\)**: The number of periods in the rolling window.

!!! important "Important"
    This estimator is **more accurate than close-to-close volatility** because it:

    ✔ Uses **high and low prices**, which contain more information than just closing prices.  
    ✔ Provides a **rolling measure of volatility**, making it suitable for **time-series analysis**.  
    ✔ Assumes that price movements follow a **Brownian motion** without drift.  

```python
def parkinson_volatility(df: pd.DataFrame, window_size: int = 30, high_col: str = 'high', low_col: str = 'low')\
                        -> pd.Series:
    """
    Calculate Parkinson's volatility.md estimator using numpy operations with Numba acceleration.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the price data.
    window_size : int, optional
        The number of periods to include in the rolling calculation (default is 30).
    high_col : str, optional
        Column name for the high prices (default is 'high').
    low_col : str, optional
        Column name for the low prices (default is 'low').

    Returns
    -------
    volatility_series : pandas.Series
        A Series indexed the same as `df`, containing the rolling Parkinson volatility.md.
        The first `window_size` rows will be NaN because there is insufficient data
        to compute the volatility.md in those windows.
    """
```

📢 *For a practical example, check out the [educational notebook](https://www.quantreo.com).*




---
### **Rogers-Satchell Volatility**
The `rogers_satchell_volatility` function computes the **Rogers-Satchell volatility**, an estimator designed specifically for **assets with a directional drift**, making it more suitable for financial markets than standard measures like **close-to-close volatility**.

!!! tip "Tip"
    Unlike standard volatility measures, the **Rogers-Satchell estimator** is a **rolling volatility** measure, meaning that at each time step, the function calculates the volatility over the **last N observations**, producing a **time series of volatility values**.

The **Rogers-Satchell estimator** is defined as:

\[
\sigma_{RS}^2 = \frac{1}{N} \sum_{i=1}^{N} \left( \ln (\frac{h_i}{c_i}) \ln (\frac{h_i}{o_i}) + \ln (\frac{l_i}{c_i}) \ln (\frac{l_i}{o_i}) \right)
\]

*(Source: Rogers & Satchell, 1994, "Estimating Variance from High, Low, and Close Prices")*

Where:

- **\(h_i\)**: The **high price** at time \(i\).
- **\(l_i\)**: The **low price** at time \(i\).
- **\(o_i\)**: The **open price** at time \(i\).
- **\(c_i\)**: The **close price** at time \(i\).
- **\(N\)**: The number of periods in the rolling window.

!!! important "Important"
    This estimator is **more accurate than standard close-to-close volatility** because it:

    ✔ Incorporates **intraday high and low prices**, providing a better estimate of actual price movement.  
    ✔ Does not assume zero drift, making it more applicable to **trending markets**.  
    ✔ Is **computationally efficient** while reducing bias in the variance estimate.  

```python
def rogers_satchell_volatility(df: pd.DataFrame, window_size: int = 30, high_col: str = 'high',
                               low_col: str = 'low', open_col: str = 'open', close_col: str = 'close') -> pd.Series:
    """
    Compute the Rogers-Satchell volatility.md estimator using a rolling window.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing OHLC price data.
    window_size : int, optional
        The number of periods used in the rolling calculation (default = 30).
    high_col : str, optional
        Column name for high prices (default = 'high').
    low_col : str, optional
        Column name for low prices (default = 'low').
    open_col : str, optional
        Column name for open prices (default = 'open').
    close_col : str, optional
        Column name for close prices (default = 'close').

    Returns
    -------
    pd.Series
        A Series containing the rolling Rogers-Satchell volatility.md, indexed like `df`.
        The first `window_size` rows are NaN due to insufficient data.
    """
```

📢 *For a practical example, check out the [educational notebook](https://www.quantreo.com).*


---

### **Yang-Zhang Volatility Estimator**

The `yang_zhang_volatility` function computes the **Yang-Zhang volatility**, an advanced estimator that integrates **overnight, close-to-close, and intra-day price movements**. This method is particularly useful for assets with **significant overnight gaps**, making it more robust than traditional measures.

!!! tip "Tip"
    Unlike standard volatility measures, the **Yang-Zhang estimator** is a **rolling volatility measure**, meaning that at each time step, the function calculates the volatility over the **last N observations**, producing a **time series of volatility values**.


The **Yang-Zhang estimator** is defined as:

\[
\sigma_t = \sqrt{\sigma_O^2 + k\sigma_C^2 + (1 - k) \sigma_{RS}^2}
\]

*(Source: Yang & Zhang, 2000, "Drift Independent Volatility Estimation")*

Where:

- **\(\sigma_O^2\)**: The **overnight volatility**, measuring the variance between **close-to-open** prices.
- **\(\sigma_C^2\)**: The **close-to-close volatility**, capturing daily price changes.
- **\(\sigma_{RS}^2\)**: The **Rogers-Satchell estimator**, which accounts for intra-day price movements.
- **\(k\)**: A weighting factor empirically determined by Yang & Zhang, according to them, the best empirical value is 0.34.

!!! important "Important"
    This estimator is **more accurate than simple close-to-close volatility** because it:

    ✔ Reduces bias from opening gaps.  
    ✔ Accounts for intra-day price fluctuations.  
    ✔ Provides a better measure of realized volatility.

```python
def yang_zhang_volatility(df: pd.DataFrame, window_size: int = 30, high_col: str = 'high', low_col: str = 'low',
                          open_col: str = 'open', close_col: str = 'close', k: float = 0.34) -> pd.Series:
    """
    Compute the Yang-Zhang volatility.md estimator using a rolling window.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing OHLC price data.
    window_size : int, optional
        The number of periods used in the rolling calculation (default = 30).
    high_col : str, optional
        Column name for high prices (default = 'high').
    low_col : str, optional
        Column name for low prices (default = 'low').
    open_col : str, optional
        Column name for open prices (default = 'open').
    close_col : str, optional
        Column name for close prices (default = 'close').
    k : float, optional
        The weighting parameter for the open-to-close variance component, as described
        in Yang & Zhang (2000). Empirical research suggests 0.34 as the optimal value.

    Returns
    -------
    pd.Series
        A Series containing the rolling Yang-Zhang volatility.md, indexed like `df`.
        The first `window_size` rows are NaN due to insufficient data.
    """

```

📢 *For a practical example, check out the [educational notebook](https://www.quantreo.com).*


---
## **Math**

``` py
from quantreo.features_engineering import math
```


---
### **Derivatives**

The `derivatives` function computes the **first and second derivatives** of a given price series, representing **velocity** (rate of change) and **acceleration** (rate of velocity change), respectively. These metrics are essential for understanding **price momentum** and **curvature** in financial time series.

#### **Mathematical Definition**
Given a price series \( P_t \):

- **First derivative (Velocity):**  Measures the rate of change of the price over time. 
  
\[
  v_t = \frac{P_{t} - P_{t-1}}{\Delta t} 
\]

- **Second derivative (Acceleration):**  Captures the curvature, indicating whether the momentum is **increasing or decreasing**.

  
\[
  a_t = \frac{v_t - v_{t-1}}{\Delta t} = \frac{P_{t} - 2P_{t-1} + P_{t-2}}{\Delta t^2}
\]

!!! tip Tip
    In practice, the function assumes $\Delta t = 1$ (e.g., one time step per observation)
    These approximations work well for discrete financial time series but may introduce noise, so **smoothing techniques** can be applied.

```python
def derivatives(df: pd.DataFrame, col: str) -> Tuple[pd.Series, pd.Series]:
    """
    Compute the first (velocity) and second (acceleration) derivatives of a specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    col : str
        The name of the column for which the derivatives are computed.

    Returns
    -------
    velocity_series : pandas.Series
        The first derivative (velocity) of the specified column.
    acceleration_series : pandas.Series
        The second derivative (acceleration) of the specified column.
    """
```
📢 *For a practical example, check out the [educational notebook](https://www.quantreo.com).*



---
### **Logarithmic Percentage Change**

The `log_pct` function computes the **log return** over a specified window. Log returns are widely used in finance as they stabilize variance and make returns **time-additive**.

\[
r_t  = \ln(P_t) - \ln(P_{t-n})
\]

Where: $P_t$ is the price at time $t$ and $n$ is the window size.

```python
def log_pct(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    """
    Compute the log-transformed percentage change over a given window.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the column to be transformed.
    col : str
        Column name to apply the transformation.
    n : int
        Window size for the percentage change.

    Returns
    -------
    pd.Series
        A Series containing the rolling log returns over `n` periods.
    """
```
📢 *For a practical example, check out the [educational notebook](https://www.quantreo.com).*


---
### **Auto Correlation**

The `auto_corr` function computes the **rolling autocorrelation** of a given column over a specified window. Autocorrelation measures how strongly a time series value is related to its **past values** at a given lag.

\[
r_k = \frac{\sum_{t=1}^{N-k} (X_t - \bar{X})(X_{t+k} - \bar{X})}{\sum_{t=1}^{N} (X_t - \bar{X})^2}
\]

Where $X_t$ is the value at time $t$, $k$ is the lag,  $N$ is the rolling window size.

```python
def auto_corr(df: pd.DataFrame, col: str, n: int = 50, lag: int = 10) -> pd.Series:
    """
    Compute the rolling autocorrelation for a specified column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data.
    col : str
        Column name to compute autocorrelation.
    n : int, optional
        Rolling window size (default = 50).
    lag : int, optional
        Lag for autocorrelation computation (default = 10).

    Returns
    -------
    pd.Series
        A Series containing the rolling autocorrelation values.
    """
```






---
### **Hurst**
The `hurst` function computes the **Hurst exponent** over a rolling window. The Hurst exponent is a **measure of long-term memory** in time series data, helping to classify a series as **mean-reverting, random, or trending**.

#### **Formula**
The Hurst exponent is estimated using **rescaled range analysis**:

\[
H = \frac{\log(R/S)}{\log(n)}
\]

Where $R$ is the range of the cumulative deviations, $S$ is the standard deviation, $n$ is the window size.

!!! tip "Tip"
    In the Quantreo's library, the Hurst exponent is a rolling measure, meaning each value represents the memory effect over the last N observations.

**Interpretation**:

- **\( H < 0.5 \)** → **Mean-reverting** (e.g., stationary processes like stock spreads).
- **\( H \approx 0.5 \)** → **Random walk** (e.g., Brownian motion, efficient markets).
- **\( H > 0.5 \)** → **Trending** (e.g., momentum-driven assets).

```python
def hurst(df: pd.DataFrame, col: str, window: int = 100) -> pd.DataFrame:
    """
    Compute the rolling Hurst exponent for a given column in a DataFrame.

    The Hurst exponent is a measure of the **long-term memory** of a time series.
    It helps determine whether a series is **mean-reverting**, **random**, or **trending**.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series data.
    col : str
        Column name on which the Hurst exponent is calculated.
    window : int, optional
        Rolling window size for the Hurst exponent computation (default = 100).

    Returns
    -------
    pd.Series
        A Series containing the rolling Hurst exponent values over the given window.
    """
```






---
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>




## Trend
## Math
## Market Regime


!!! tip "Tip"
    If you don't have prior experience with Python, we recommend reading [Using Python's pip to Manage Your Projects' Dependencies](https://pip.pypa.io/en/stable/user_guide/), which is a really good introduction on the mechanics of Python package management and helps you troubleshoot if you run into errors.


!!! danger "Danger"
    Running this command may delete important files!

!!! note "Note"
    This is a simple note box.

!!! important "Important"
    Make sure you install all dependencies before proceeding.

