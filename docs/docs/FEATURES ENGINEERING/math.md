# **Math**

``` py
from quantreo.features_engineering import math
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
## **Logarithmic Percentage Change**

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
## **Auto Correlation**

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
