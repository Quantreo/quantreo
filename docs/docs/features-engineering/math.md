# **Math**
You can find a series of examples on how to create these features in the [educational notebooks](/../tutorials/features-engineering-math) provided by Quantreo.

``` py
import quantreo.features_engineering as fe
```


---
## **Derivatives**

The `derivatives` function computes the **first and second derivatives** of a given price series, representing **velocity** (rate of change) and **acceleration** (rate of velocity change), respectively. These metrics are essential for understanding **price momentum** and **curvature** in financial time series.

<br>
**Mathematical Definition**

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

=== "Function"
    ```python
    def derivatives(df: pd.DataFrame, col: str) -> Tuple[pd.Series, pd.Series]
    ```
=== "Docstring"
    ```python
    """
    Compute the first (velocity) and second (acceleration) derivatives of a specified column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    col : str
        The name of the column for which the derivatives are computed.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        velocity_series : first derivative (velocity)
        acceleration_series : second derivative (acceleration)
    """
    ```
=== "Example"
    ```python
    df["velocity"], df["acceleration"] = derivatives(df=df, col="close")
    ```
📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#derivatives).*



---
## **Logarithmic Percentage Change**

The `log_pct` function computes the **log return** over a specified window. Log returns are widely used in finance as they stabilize variance and make returns **time-additive**.

\[
r_t  = \ln(P_t) - \ln(P_{t-n})
\]

Where: $P_t$ is the price at time $t$ and $n$ is the window size.

=== "Function"
    ```python
    def log_pct(df: pd.DataFrame, col: str, window_size: int) -> pd.Series
    ```
=== "Docstring"
    ```python
    """
    Compute the logarithmic percentage change (log return) over a specified window.

    Mathematically:
        log_pct[t] = log(P_t) - log(P_{t - window_size})
                   = log(P_t / P_{t - window_size})

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the column to be analyzed.
    col : str
        The name of the column containing price or value data.
    window_size : int
        The number of periods over which to compute the log return.

    Returns
    -------
    pd.Series
        A Series containing the rolling log returns over `window_size` periods.
    """
    ```
=== "Example"
    ```python
    df["log_pct"] = log_pct(df=df, col="close", window_size=30)
    ```


📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#logarithmic-percentage-change).*


---
## **Auto Correlation**

The `auto_corr` function computes the **rolling autocorrelation** of a given column over a specified window. Autocorrelation measures how strongly a time series value is related to its **past values** at a given lag.

\[
r_k = \frac{\sum_{t=1}^{N-k} (X_t - \bar{X})(X_{t+k} - \bar{X})}{\sum_{t=1}^{N} (X_t - \bar{X})^2}
\]

Where $X_t$ is the value at time $t$, $k$ is the lag,  $N$ is the rolling window size.


=== "Function"
    ```python
    def auto_corr(df: pd.DataFrame, col: str, window_size: int = 50, lag: int = 10) -> pd.Series
    ```
=== "Docstring"
    ```python
    """
    Compute rolling autocorrelation for a specified column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name to compute autocorrelation.
    window_size : int, optional
        Rolling window size (default=50).
    lag : int, optional
        Lag for autocorrelation (default=10).

    Returns
    -------
    pd.Series
        Rolling autocorrelation values.
    """
    ```
=== "Example"
    ```python
    df["auto_corr"] = auto_corr(df=df, col="close", window_size=50, lag=10)
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

**Interpretation**:

- **\( H < 0.5 \)** → **Mean-reverting** (e.g., stationary processes like stock spreads).
- **\( H \approx 0.5 \)** → **Random walk** (e.g., Brownian motion, efficient markets).
- **\( H > 0.5 \)** → **Trending** (e.g., momentum-driven assets).


=== "Function"
    ```python
    def hurst(df: pd.DataFrame, col: str, window_size: int = 100) -> pd.Series
    ```
=== "Docstring"
    ```python
    """
    Compute rolling Hurst exponent to assess long-term memory in a series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column to analyze.
    window_size : int, default=100
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling Hurst exponent.
    """
    ```
=== "Example"
    ```python
    df["hurst"] = hurst(df=df, col="close", window_size=200)
    ```
=== "Notes"
    !!! tip "Tip"
        In the Quantreo's library, the Hurst exponent is a rolling measure, meaning each value represents the memory effect over the last N observations.


📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#hurst).*

---
## **Augmented Dickey–Fuller (ADF)**  

The `adf_test` function measures **stationarity** in a time‑series via a rolling Augmented Dickey–Fuller test.  
Stationarity is crucial for many models (ARIMA, pairs‑trading, mean‑reversion signals).

**Regression Types**

| `regression` | Component(s) included | Typical use‑case |
|--------------|----------------------|------------------|
| `"c"`        | constant only        | detect stationarity around a *fixed mean* |
| `"ct"`       | constant + trend     | detect stationarity around a *linear trend* |

**Interpretation**

- **p‑value ≪ 0.05**  →  *Reject* the unit‑root null ⇒ series is **stationary** in that window.  
- **p‑value ≈ 1**     →  cannot reject null ⇒ behaves like a **random walk**.  
- Monitor the rolling statistic (*adf_stat*) to see how strongly the unit‑root hypothesis is rejected (more negative ⇒ stronger evidence of stationarity).

=== "Function"
    ```python
    def adf_test(df: pd.DataFrame, col: str, window_size: int, lags: int | None = None, regression: str = "c") -> Tuple[pd.Series, pd.Series]
    ```

=== "Docstring"
    ```python
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

=== "Example"
    ```python
    df["adf_stat"], df["adf_pval"] = adf_test(df=df, col="close", window_size=80, lags=10, regression="ct")
    ```

=== "Notes"
    !!! tip "Tip"
        In Quantreo, ADF is **rolling**, each value tells you how stationary the *last N* observations are.


📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#adf-test).*


---
## **ARCH (Engle) Test**

The `arch_test` function detects **conditional heteroskedasticity**, volatility clustering, by applying Engle’s ARCH Lagrange‑Multiplier test on a **rolling window**.

### Interpretation

- **Low p‑value (< 0.05)** ⇒ Reject homoskedasticity ⇒ **volatility is clustered** in that window.  
- The raw `arch_stat` (LM) grows with the strength of clustering and with the window length.


=== "Function"
    ```python
    def arch_test(df: pd.DataFrame, col: str, window_size: int = 60, lags: int = 5, ddof: int = 0) -> Tuple[pd.Series, pd.Series]
    ```

=== "Docstring"
    ```python
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

=== "Example"
    ```python
    df["arch_stat"], df["arch_pval"] = arch_test(df=df, col="returns", window_size=60, lags=10)
    ```
=== "Notes"
    !!! tip "Tip"
        In Quantreo, ARCH is **rolling**, each value tells you the ARCH stat (or p-value) over the *last N* observations.


📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#arch-test).*


---
## **Skewness**

The `skewness` function captures **asymmetry** in the return distribution over a rolling window.  
Skewness is useful for detecting *tail‑risk*: strong negative skew hints at infrequent but severe losses, while positive skew indicates the potential for rare large gains.

**Quick interpretation**
- **Positive skew (`> 0`)**: Long right tail. Indicates occasional outsized gains or extreme positive returns.
- **Symmetric (`≈ 0`)**: Balanced distribution. Behavior close to Gaussian, no major bias in direction.
- **Negative skew (`< 0`)**: Long left tail. Indicates crash-prone dynamics or fat left tail (extreme losses).

=== "Function"
    ```python
    def skewness(df: pd.DataFrame, col: str, window_size: int = 60) -> pd.Series
    ```

=== "Docstring"
    ```python
    """
    Compute rolling skewness (third standardized moment) of a time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series.
    col : str
        Column to compute skewness on.
    window_size : int, optional
        Rolling window size (default=60).

    Returns
    -------
    pd.Series
        Rolling skewness of the specified column.
    """
    ```

=== "Example"
    ```python
    df["skewness"] = skewness(df=df, col="returns", window_size=60)
    ```
=== "Notes"
    !!! tip "Tip"
        Quantreo computes skewness **rolling**, each value summarises the asymmetry of the *last N* observations.


📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#skewness).*

---
## **Kurtosis**

The `kurtosis` function tracks **tail heaviness** in returns over a rolling window.  
High kurtosis warns of extreme moves (fat tails); low kurtosis indicates a thinner‑tailed, more Gaussian‑like distribution.

**Quick interpretation**

- **High kurtosis (`> 3`)**: Fat tails.Large shocks are more likely (extreme returns, volatility clustering).
- **Normal kurtosis (`≈ 3`)**: Gaussian (normal distribution). Standard volatility, typical behavior.
- **Low kurtosis (`< 3`)**: Light tails. Less prone to outliers or sharp price changes.

=== "Function"
    ```python
    def kurtosis(df: pd.DataFrame, col: str, window_size: int = 60) -> pd.Series
    ```

=== "Docstring"
    ```python
    """
    Compute the rolling kurtosis (fourth standardized moment) of a time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series.
    col : str
        Column to compute kurtosis on.
    window_size : int, optional
        Rolling window size (default=60).

    Returns
    -------
    pd.Series
        Rolling kurtosis of the specified column.
    """
    ```

=== "Example"
    ```python
    df["kurtosis"] = kurtosis(df=df, col="returns", window_size=60)
    ```

=== "Notes"
    !!! tip "Tip"
        Quantreo’s kurtosis is **rolling**, each value reflects tail risk across the *last N* observations.


📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#kurtosis).*

---

## **Bimodality Coefficient**

The `bimodality_coefficient` function computes the **Bimodality Coefficient (BC)**, a statistical measure used to detect whether a distribution is **unimodal** (single regime) or **bimodal/multimodal** (multiple regimes).  

It combines **skewness ($γ$)** and **kurtosis ($κ$)** into a single rolling indicator.  
A BC greater than **0.55** typically indicates the presence of bimodality or regime-switching behavior.

The formula is:

$$
BC = \frac{γ^2 + 1}{κ + \frac{3(n-1)^2}{(n-2)(n-3)}}
$$

- $γ$ = skewness  
- $κ$ = excess kurtosis (normal distribution = 0)
- $n$ = rolling window size  

<br>

**Typical use-cases** 

- Detecting **market regime shifts** (calm vs volatile regimes).  
- Identifying **mixture distributions** in returns or volatility.  
- Pre-filtering signals for **regime-dependent strategies**.  


!!! tip "Interpretation"
    - **BC < 0.55** → distribution is likely **unimodal** (stable regime).  
    - **BC > 0.55** → distribution is likely **bimodal/multimodal** (two or more regimes coexisting).  


=== "Function"
    ```python
    def bimodality_coefficient(df: pd.DataFrame, col: str, window_size: int = 100) -> pd.Series
    ```

=== "Docstring"
    ```python
    """
    Compute the rolling Bimodality Coefficient (BC).

    BC quantifies whether a distribution is unimodal (single regime)
    or bimodal/multimodal (multi-regime). A BC > 0.55 typically
    indicates regime switching or bimodality.

    Formula:
        BC = (γ² + 1) / (κ + 3*(n−1)² / ((n−2)*(n−3)))

    where:
        γ = skewness
        κ = excess kurtosis (normal = 0)
        n = window size

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data.
    col : str
        Column to compute BC on (e.g. returns).
    window_size : int, optional
        Rolling window size (default=100).

    Returns
    -------
    pd.Series
        Rolling Bimodality Coefficient values.
    """
    ```

=== "Example"
    ```python
    df["bimodality"] = bimodality_coefficient(df=df, col="returns", window_size=250)
    ```

📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#bimodality-coefficient).*


---
## Sample Entropy

Compute the rolling Sample Entropy of a time series column. Sample Entropy is a non-linear measure of irregularity or unpredictability in a signal.  
It helps detect transitions between more structured and more chaotic market conditions.

!!! tip "Understanding the 'order' parameter"
    The `order` parameter determines how many consecutive observations are grouped to form each pattern before computing entropy.  
    For example, `order = 2` means the algorithm looks at patterns of 2 consecutive values (e.g., `[x₁, x₂]`, `[x₂, x₃]`, etc.),  
    while `order = 3` forms patterns of 3 values. Higher orders allow detection of more complex structures,  
    but require more data and are more sensitive to noise.

<br>

**Interpretation**: A higher Sample Entropy value indicates a more irregular, unpredictable, and chaotic time series. Lower values suggest more structured or repetitive patterns in the data.


=== "Function"
    ```python
    def sample_entropy(df: pd.DataFrame, col: str = "close", window_size: int = 60, order: int = 2) -> pd.Series
    ```

=== "Docstring"
    ```python
    """
    Compute the rolling Sample Entropy of a time series.

    Sample Entropy quantifies the level of irregularity or unpredictability
    in a signal. It helps detect transitions between structured and chaotic
    market conditions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series.
    col : str, optional
        Column to compute entropy on (default = "close").
    window_size : int, optional
        Rolling window size (default = 60).
    order : int, optional
        Embedding dimension for pattern comparison (default = 2).

    Returns
    -------
    pd.Series
        Rolling Sample Entropy values. The first (window_size - 1) values will be NaN.
    """
    ```

=== "Example"
    ```python
    df["sample_entropy"] = sample_entropy(df=df, col="close", window_size=60, order=3)
    ```

📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#sample-entropy).*

---

## Spectral Entropy

Compute the rolling Spectral Entropy of a time series column.  
Spectral Entropy is a frequency-domain measure that quantifies how the power of a signal is distributed across frequency components.  
It helps detect whether a signal is highly structured (low entropy) or spread across many frequencies (high entropy).

<br>


**Interpretation**: 

- A **high spectral entropy** means the signal's energy is spread across many frequencies → more chaotic and less structured.  
- A **low value** suggests the signal is concentrated in a few frequencies → more regular or predictable.

=== "Function"

    ```python
    fe.math.spectral_entropy(df: pd.DataFrame, col: str = "close", window_size: int = 100, sf: int = 1,
                                 method: str = 'welch', normalize: bool = True, nperseg: int = None) -> pd.Series
    ```
=== "Docstring"
    ```python
    """
    Calculate the rolling Spectral Entropy of a time series.
    
    Spectral Entropy quantifies the flatness or complexity of the power
    spectral density of a signal. It provides insight into the frequency
    content and structure of a time series.
    
    This function applies spectral entropy over a rolling window, allowing
    dynamic tracking of complexity in the frequency domain.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series.
    col : str, default="close"
        The name of the column on which to compute the entropy.
    window_size : int, default=100
        Size of the rolling window (must be >= 16).
    sf : int, default=1
        Sampling frequency used in spectral estimation (must be > 0).
    method : str, default="welch"
        Method used to compute the power spectral density ("welch" or "fft").
    normalize : bool, default=True
        Whether to normalize entropy to [0, 1].
    nperseg : int, optional
        Segment length for Welch's method. If None, defaults to min(window_size, window_size // 2).
    
    Returns
    -------
    pd.Series
        A Series containing the rolling Spectral Entropy values. The first
        (window_size - 1) values will be NaN.
    """
    ```
=== "Example"
    ```python
    df["spectral_entropy"] = spectral_entropy(df=df, col="close", window_size=100)
    ```

📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#spectral-entropy).*

---
## Permutation Entropy

Compute the rolling Permutation Entropy of a time series column.  
Permutation Entropy is a non-linear complexity measure based on the order relationships between time series values.  
It is particularly effective at detecting subtle structural changes or dynamic regime shifts in financial data.

<br>


**Interpretation**:

- A **high permutation entropy** means that the ordering of values is highly irregular and chaotic.
- A **low value** indicates more structured or repeated patterns in the signal.

=== "Function"
    
    ```python
    fe.math.permutation_entropy(df: pd.DataFrame, col: str = "close", window_size: int = 100, order: int = 3,
                                    delay: int = 1, normalize: bool = True) -> pd.Series
    ```

=== "Docstring"
    
    ```python
    """
    Calculate the rolling Permutation Entropy of a time series.
    
    Permutation Entropy quantifies the complexity of temporal ordering in a signal.
    It is particularly useful for detecting subtle dynamic changes in structure.
    
    This function computes Permutation Entropy over a sliding window,
    providing a real-time view of signal complexity.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series.
    col : str, default="close"
        The name of the column on which to compute the entropy.
    window_size : int, default=100
        Size of the rolling window (must be >= 10).
    order : int, default=3
        Embedding dimension for permutation patterns (must be >= 2).
    delay : int, default=1
        Time delay between points used in embedding (must be >= 1).
    normalize : bool, default=True
        Whether to normalize entropy to [0, 1].
    
    Returns
    -------
    pd.Series
        A Series containing the rolling Permutation Entropy values.
        The first (window_size - 1) values will be NaN.
    """
    ```

=== "Example"
    ```python
    df["perm_entropy"] = permutation_entropy(df=df, col="close", window_size=100, order=3)
    ```

📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#permutation-entropy).*

---
## Detrended Fluctuation Analysis (DFA)

Compute the rolling Detrended Fluctuation Analysis (DFA) exponent of a time series column.  
DFA is a fractal analysis method that quantifies long-term memory and self-similarity in a signal.  
It is useful for detecting **persistence** (trend-following behavior) or **anti-persistence** (mean-reversion behavior) in financial regimes.

<br>


**Interpretation**:

- **DFA ≈ 0.5** → random walk (white noise)  
- **DFA > 0.5** → persistent behavior (trend continuation)  
- **DFA < 0.5** → anti-persistent behavior (mean-reversion)

=== "Function"
    ```python
    fe.math.detrended_fluctuation(df: pd.DataFrame,  col: str = "close", window_size: int = 100, nvals: np.ndarray = None,
                              overlap: bool = False) -> pd.Series
    ```

=== "Docstring"
    ```python
    """
    Calculate the rolling Detrended Fluctuation Analysis (DFA) exponent of a time series.
    
    DFA measures long-term memory and fractal scaling in a time series,
    making it suitable for detecting persistence or anti-persistence in market regimes.
    
    This function applies DFA over a rolling window, producing a time-varying
    indicator of signal regularity and self-similarity.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series.
    col : str, default="close"
        The name of the column on which to compute the DFA exponent.
    window_size : int, default=100
        Size of the rolling window (must be >= 100).
    nvals : np.ndarray, optional
        Array of segment sizes. If None, defaults to logspace from 4 to N/4.
    overlap : bool, default=False
        Whether to use overlapping windows when computing fluctuations.
    
    Returns
    -------
    pd.Series
        A Series containing the rolling DFA exponents.
        The first (window_size - 1) values will be NaN.
    """
    ```

=== "Example"
    ```python
    df["dfa"] = detrended_fluctuation(df=df, col="close", window_size=100)
    ```

📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#detrend-fluctuation).*


---
## Petrosian Fractal Dimension (PFD)

Compute the rolling Petrosian Fractal Dimension of a time series column. Petrosian FD estimates the structural complexity of a signal by measuring directional changes.  
It is fast to compute and useful for capturing signal regularity.

<br>

**Interpretation** - In most practical applications (financial time series or EEG signals), typical PFD values fall within these ranges:

- **1.00–1.05** → very regular signal (quasi-linear or simple structure)  
- **1.05–1.10** → moderately irregular signal (structured noise, moderate volatility)  
- **1.10+** → highly irregular signal, high-frequency noise, or chaotic behavior

=== "Function"
    ```python
    fe.math.petrosian_fd(df: pd.DataFrame, col: str = "close", window_size: int = 100) -> pd.Series
    ```

=== "Docstring"
    ```python
    """
    Calculate the rolling Petrosian Fractal Dimension (FD) of a time series.
    
    Petrosian FD measures the structural complexity of a signal based on
    changes in the direction of the signal's first derivative.
    
    This function applies the Petrosian FD over a rolling window,
    producing a time series that tracks signal complexity in real-time.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series.
    col : str, default="close"
        The name of the column on which to compute the fractal dimension.
    window_size : int, default=100
        Size of the rolling window (must be >= 10).
    
    Returns
    -------
    pd.Series
        A Series containing the rolling Petrosian FD values.
        The first (window_size - 1) values will be NaN.
    """
    ```

=== "Example"
    ```python
    df["petrosian_fd"] = petrosian_fd(df=df, col="close", window_size=100)
    ```


📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#petrosian-fd).*


---
## Tail Index (Hill estimator)

Compute the rolling tail index (α̂) using the Hill estimator (without the +1 bias correction) on the **right tail** of a strictly positive time series.

This method captures the **heaviness of the tail distribution**:  
lower values of α indicate fatter tails (more probability of extreme events),  
while higher α implies a thinner tail and less extreme behavior.

!!! danger "**Important**: Only Positive Values"
    This estimator works **only** on strictly positive values.  
    You must pre-process your series (e.g., `abs(returns)` or `-returns[returns < 0]`) depending on the tail you want to study.


**Interpretation**:

- **α̂ < 2** → Very heavy-tailed distribution → Extreme moves likely  
- **α̂ ≈ 3** → Comparable to a t-distribution (df=3) → High volatility, finite variance  
- **α̂ > 4** → Thinner tails → Lower probability of rare events

=== "Function"
    ```python
    fe.math.tail_index(df: pd.DataFrame, col: str = "close", window_size: int = 250, k_ratio: float = 0.10) -> pd.Series
    ```
=== "Example"
    ```python
    """
    Rolling Hill tail‑index (α̂, *without* the +1 bias‑correction).
    
    *Right‑tail* estimator – **`df[col]` must contain strictly positive values**
    (e.g. absolute returns, drawdown magnitudes).
    Any window that includes ≤ 0 is skipped.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data frame.
    col : str, default "close"
        Column on which to compute α̂(t).
    window_size : int, default 250
        Rolling window length *n*.
    k_ratio : float, default 0.10
        Fraction of the window regarded as the tail
        (`k = max(1, int(round(k_ratio * window_size)))`).
        5 – 15 % is a common compromise between bias and variance.
    
    Returns
    -------
    pd.Series
        α̂(t) aligned with `df.index`; the first `window_size−1` points are `NaN`.
    
    """
    ```

=== "Example"
    ```python
    df["abs_returns"] = np.abs(df["returns"])
    df["tail_index"] = tail_index(df=df, col="abs_returns", window_size=150, k_ratio=0.10)
    ```


📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#tail-index).*


---
## Shapiro Wilk

Applies the Shapiro-Wilk test on a time series using a rolling window to evaluate the local normality of a specified column.

This test detects deviations from normality by capturing both skewness and kurtosis. It is especially useful for identifying statistical regime changes in financial return series.

<br>

**Interpretation**

- **p-value < 0.05** → Reject H₀: the distribution is **not normal**  
- **p-value ≥ 0.05** → Fail to reject H₀: the distribution **may be normal**  

=== "Function"
    ```python
    fe.math.shapiro_wilk(df: pd.DataFrame, col: str, window_size: int) -> tuple[pd.Series, pd.Series]
    ```

=== "Docstring"
    ```python
    """
    Rolling Shapiro-Wilk test for normality on a time series column.
    
    This function evaluates the null hypothesis that the data in the specified column 
    comes from a normal distribution. It applies the test over a rolling window 
    of fixed size and returns both the test statistic (W) and the associated p-value 
    at each time step.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series.
    col : str
        Name of the column to test for normality.
    window_size : int
        Rolling window size.
    
    Returns
    -------
    stat_series : pd.Series
        Series of W statistics from the Shapiro-Wilk test.
    pval_series : pd.Series
        Series of p-values corresponding to each window.
    """
    ```

=== "Example"
    ```python
    df["sw_stat"], df["sw_pvalue"] = shapiro_wilk(df=df, col="returns", window_size=60)
    ```


📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-math/#shapiro-wilk).*