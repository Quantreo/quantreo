# **Math**
You can find a series of examples on how to create these features in the [educational notebooks](/../tutorials/features-engineering-math) provided by Quantreo.

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
📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#derivatives).*



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
📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#logarithmic-percentage-change).*


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
📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#auto-correlation).*






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
📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#hurst).*

---
## **Augmented Dickey–Fuller (ADF)**  

The `adf_test` function measures **stationarity** in a time‑series via a rolling Augmented Dickey–Fuller test.  
Stationarity is crucial for many models (ARIMA, pairs‑trading, mean‑reversion signals).

!!! tip "Tip"
    In Quantreo, ADF is **rolling** – each value tells you how stationary the *last N* observations are.

**Regression flavours**

| `regression` | Component(s) included | Typical use‑case |
|--------------|----------------------|------------------|
| `"c"`        | constant only        | detect stationarity around a *fixed mean* |
| `"ct"`       | constant + trend     | detect stationarity around a *linear trend* |

**Interpretation**

- **p‑value ≪ 0.05**  →  *Reject* the unit‑root null ⇒ series is **stationary** in that window.  
- **p‑value ≈ 1**     →  cannot reject null ⇒ behaves like a **random walk**.  
- Monitor the rolling statistic (*adf_stat*) to see how strongly the unit‑root hypothesis is rejected (more negative ⇒ stronger evidence of stationarity).

```python title="How to call the adf_test function"
fe.math.adf_test(df: pd.DataFrame, col: str, window_size: int, lags: int | None = None, regression: str = "c")
```

``` title="adf_test function docstring"
"""
Compute the Augmented Dickey-Fuller test in rolling windows to estimate stationarity over time.

This function applies the ADF test in rolling fashion to a given column of a DataFrame.
You can choose between a constant-only regression ('c') or a constant + linear trend ('ct').
The p-values are approximated using fast interpolated tables, avoiding `statsmodels` overhead.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the time series to analyze.
col : str
    Name of the column to test for stationarity.
window_size : int
    Size of the rolling window to compute the ADF test.
lags : int, optional (default=None)
    Number of lagged differences to include in the regression. If None, uses Schwert's rule.
regression : str, optional (default='c')
    Type of regression to run:
    - 'c'  : constant only (tests stationarity around a non-zero mean)
    - 'ct' : constant + trend (tests stationarity around a linear trend)

Returns
-------
tuple[pd.Series, pd.Series]
    - ADF statistic for each rolling window
    - Corresponding interpolated p-values
"""
```
📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#adf-test).*


---
## **ARCH (Engle) Test**

The `arch_test` function detects **conditional heteroskedasticity** — volatility clustering — by applying Engle’s ARCH Lagrange‑Multiplier test on a **rolling window**.

!!! tip "Tip"
    In Quantreo, ARCH is **rolling** – each value tells you the ARCH stat (or p-value) over the *last N* observations.

### Interpretation

- **Low p‑value (< 0.05)** ⇒ Reject homoskedasticity ⇒ **volatility is clustered** in that window.  
- The raw `arch_stat` (LM) grows with the strength of clustering and with the window length.

```python title="How to call the arch_test function"
fe.math.arch_test(df: pd.DataFrame, col: str, window_size: int = 60, lags: int | None = 5, ddof: int = 0)
```

``` title="arch_test function docstring"
"""
Compute the ARCH test (Engle) over rolling windows to detect conditional heteroskedasticity.

This function applies the ARCH Lagrange Multiplier test in a rolling fashion
to a given time series. It returns both the LM statistic and the associated p-value.
The ARCH test measures whether volatility is autocorrelated (i.e., clustering),
which is common in financial time series.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the time series data.
col : str
    Name of the column to test (typically returns or residuals).
window_size : int, optional (default=60)
    Size of the rolling window used to estimate ARCH effects.
lags : int, optional (default=5)
    Number of lags to include in the ARCH regression (squared residuals).
ddof : int, optional (default=0)
    Degrees of freedom adjustment (useful when residuals come from a fitted model).

Returns
-------
arch_stat : pd.Series
    Rolling series of the LM statistics from the ARCH test.
arch_pval : pd.Series
    Rolling series of the associated p-values (under Chi2 distribution).

Raises
------
ValueError
    If inputs are invalid: missing column, non-numeric data, or incorrect parameters.
"""
```
📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#arch-test).*


---
## **Skewness**

The `skewness` function captures **asymmetry** in the return distribution over a rolling window.  
Skewness is useful for detecting *tail‑risk*: strong negative skew hints at infrequent but severe losses, while positive skew indicates the potential for rare large gains.

!!! tip "Tip"
    Quantreo computes skewness **rolling** – each value summarises the asymmetry of the *last N* observations.

**Quick interpretation**
- **Positive skew (`> 0`)**: Long right tail. Indicates occasional outsized gains or extreme positive returns.
- **Symmetric (`≈ 0`)**: Balanced distribution. Behavior close to Gaussian, no major bias in direction.
- **Negative skew (`< 0`)**: Long left tail. Indicates crash-prone dynamics or fat left tail (extreme losses).

```python title="How to call skewness"
fe.math.skewness(df: pd.DataFrame, col: str, window_size: int = 60)
```
``` title="skewness function docstring"
"""
    Compute the skewness (third standardized moment) over a rolling window.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series data.
    col : str
        Name of the column to compute skewness on.
    window_size : int, optional (default=60)
        Number of periods for the rolling window.

    Returns
    -------
    pd.Series
        Rolling skewness of the specified column.
"""
```
📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#skewness).*

---
## **Kurtosis**

The `kurtosis` function tracks **tail heaviness** in returns over a rolling window.  
High kurtosis warns of extreme moves (fat tails); low kurtosis indicates a thinner‑tailed, more Gaussian‑like distribution.

!!! tip "Tip"
    Quantreo’s kurtosis is **rolling** – each value reflects tail risk across the *last N* observations.

**Quick interpretation**

- **High kurtosis (`> 3`)**: Fat tails.Large shocks are more likely (extreme returns, volatility clustering).
- **Normal kurtosis (`≈ 3`)**: Gaussian (normal distribution). Standard volatility, typical behavior.
- **Low kurtosis (`< 3`)**: Light tails. Less prone to outliers or sharp price changes.

```python title="How to call kurtosis"
fe.math.kurtosis(df: pd.DataFrame, col: str, window_size: int = 60)
```

``` title="kurtosis function docstring"
"""
Compute the kurtosis (fourth standardized moment) over a rolling window.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the time series data.
col : str
    Name of the column to compute kurtosis on.
window_size : int, optional (default=60)
    Number of periods for the rolling window.

Returns
-------
pd.Series
    Rolling kurtosis of the specified column.
```
📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#kurtosis).*
