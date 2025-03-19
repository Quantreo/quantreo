# **Math**

``` py
import quantreo.features_engineering as fe
```


---
## **Derivatives**

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

```python title="How to call the derivatives function"
fe.math.derivatives(df: pd.DataFrame, col: str)
```
``` title="derivatives function docstring"
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
## **Logarithmic Percentage Change**

The `log_pct` function computes the **log return** over a specified window. Log returns are widely used in finance as they stabilize variance and make returns **time-additive**.

\[
r_t  = \ln(P_t) - \ln(P_{t-n})
\]

Where: $P_t$ is the price at time $t$ and $n$ is the window size.

```python title="How to call the logarithmic percentage function"
fe.math.log_pct(df: pd.DataFrame, col: str, window_size: int)
```
```title="loc_pct function docstring"

"""
Apply a logarithmic transformation to a specified column in a DataFrame and calculate
the percentage change of the log-transformed values over a given window size.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the column to be logarithmically transformed.
col : str
    The name of the column to which the logarithmic transformation is applied.
window_size : int
    The window size over which to calculate the percentage change of the log-transformed values.

Returns
-------
pd.Series
    A Series containing the rolling log returns over `window_size` periods.
"""
```
📢 *For a practical example, check out the [educational notebook](https://www.quantreo.com).*


---
## **Auto Correlation**

The `auto_corr` function computes the **rolling autocorrelation** of a given column over a specified window. Autocorrelation measures how strongly a time series value is related to its **past values** at a given lag.

\[
r_k = \frac{\sum_{t=1}^{N-k} (X_t - \bar{X})(X_{t+k} - \bar{X})}{\sum_{t=1}^{N} (X_t - \bar{X})^2}
\]

Where $X_t$ is the value at time $t$, $k$ is the lag,  $N$ is the rolling window size.

```python title="How to call the Auto Correlation function"
fe.math.auto_corr(df: pd.DataFrame, col: str, window_size: int = 50, lag: int = 10)
```

```title="auto_corr function docstring"
"""
Compute the rolling autocorrelation for a specified column.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the data.
col : str
    Column name to compute autocorrelation.
window_size : int, optional
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
## **Hurst**
The `hurst` function computes the **Hurst exponent** over a rolling window. The Hurst exponent is a **measure of long-term memory** in time series data, helping to classify a series as **mean-reverting, random, or trending**.

**Formula**:

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

```python title="How to call the Hurst function"
fe.math.hurst(df: pd.DataFrame, col: str, window_size: int = 100)
```

``` title="hurst function docstring"
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
window_size : int, optional
    Rolling window size for the Hurst exponent computation (default = 100).

Returns
-------
pd.Series
    A Series containing the rolling Hurst exponent values over the given window.
"""
```
