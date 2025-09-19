# **Transformation**

The `transformation` module contains functions that **reshape, rescale, or smooth raw price data** to make it more informative or statistically stable before further analysis.

You can find multiple examples of how to apply these transformations in the [educational notebooks](/../tutorials/features-engineering-transformation) provided by Quantreo.


``` py
import quantreo.features_engineering as fe
```

---
## **Fisher Transform**

The `fisher_transform` function applies a mathematical transformation that converts **non-Gaussian price data into a distribution resembling a normal curve**.

This is particularly useful for **highlighting turning points**, because values tend to cluster near the center and **amplify at the extremes** (±2 or more), making overbought and oversold conditions easier to detect.

The formula is:

\[
x_t = \frac{2 \cdot \left( \text{median}_t - \min_t \right)}{\max_t - \min_t} - 1
\quad \Rightarrow \quad
\text{Fisher}_t = 0.5 \cdot \ln\left( \frac{1 + x_t}{1 - x_t} \right)
\]

- `x_t` is a normalized representation of the price at time `t`, scaled between -1 and 1.
- The Fisher Transform **amplifies edge behavior**, making it a good signal enhancer.

```python title="How to call fisher_transform"
fe.transformation.fisher_transform(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low', window_size: int = 10)
```

``` title="fisher_transform function docstring"
"""
Compute the Fisher Transform indicator.

The Fisher Transform maps price data into a Gaussian-like distribution
using the formula:
    Fisher = 0.5 * ln((1 + x) / (1 - x))

Where:
    x = 2 * (median - min) / (max - min) - 1

It is typically used to detect turning points and overbought/oversold conditions.

Args:
    df (pd.DataFrame): DataFrame containing OHLC price data.
    high_col (str): Column name for the high price.
    low_col (str): Column name for the low price.
    window_size (int): Rolling window used to normalize the price range.

Returns:
    pd.Series: A Series containing the Fisher Transform values.
"""
```

📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-transformation/#fisher-transform)."

---

## **Savitzky-Golay Filter (Causal)**

The `savgol_filter` function applies a **causal version of the Savitzky-Golay filter** to a time series.

This transformation fits a **polynomial of order `k`** over a **rolling window** of past values and evaluates the value of that polynomial at the **last point** of the window. It is widely used to **smooth noisy signals** while preserving **local structure** like peaks and inflection points.

!!! warning "Causal vs Standard Savitzky-Golay"

    This implementation differs from the classic `scipy.signal.savgol_filter`, which applies a **centered filter** that includes both past and future values — causing **look-ahead bias** in real-time trading.
    
    Here, the filter is applied in a **rolling**, meaning it only uses **past observations** within the window.  
    This makes it **safe for backtesting and live usage**, while preserving most of the smoothing benefits of the standard version.


```python title="How to call savgol_filter"
fe.transformation.savgol_filter(df: pd.DataFrame,  col: str = 'close', window_size: int = 11, polyorder: int = 3)
```

``` title="savgol_filter function docstring"
"""
Compute a causal Savitzky-Golay filter using optimized matrix operations.

This version reproduces the behavior of:
    df[col].rolling(window_size).apply(lambda x: savgol_filter(x, window_size, polyorder)[-1])

It applies a rolling polynomial regression over a past-only window, avoiding look-ahead bias.

Args:
    df (pd.DataFrame): DataFrame containing the input series.
    col (str): Column name of the series to smooth.
    window_size (int): Length of the rolling window (must be odd).
    polyorder (int): Degree of the fitted polynomial (must be < window_size).

Returns:
    pd.Series: Smoothed series using the causal Savitzky-Golay filter.
"""
```
📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-transformation/#savitzky-golay-filter)."


---
## **Logit**

The `logit_transform` applies the **log-odds** mapping to a feature bounded in **(0, 1)**.  
It stretches values near 0 and 1 (often the most informative) and maps (0, 1) to **(−∞, +∞)**.

$$
\operatorname{logit}(x)=\ln\!\left(\frac{x}{1-x}\right),\qquad x\in(0,1)
$$

**Typical use-cases**

- Probabilities / classification scores  
- Bounded indicators rescaled to (0, 1) (e.g., normalized features, CDF outputs)  
- Emphasizing extremes while making odds additive

!!! tip "Numerical stability"
    Exact 0 or 1 yield infinities. Quantreo **clips** to $[\,\varepsilon,\ 1-\varepsilon\,]$ (default $\varepsilon=10^{-6}$).  
    After logit, values are unbounded, consider a robust scaler for downstream models.

```python title="How to call the logit_transform function"
fe.transformations.non_linear.logit_transform(df: pd.DataFrame, col: str, eps: float = 1e-6)
```

```python title="logit_transform function docstring"
"""
Apply the logit (log-odds) transformation to a column bounded in (0, 1).

The logit is defined as:
    logit(x) = log(x / (1 - x))

To avoid infinities when x is 0 or 1, values are clipped to [eps, 1 - eps].

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the data.
col : str
    Column name on which to apply the transformation. Values should lie in (0, 1).
eps : float, optional (default=1e-6)
    Small positive value used for clipping to prevent numerical issues.

Returns
-------
pd.Series
    Transformed series with the same index and name `{col}_logit`.

Raises
------
ValueError
    If `col` does not exist or is not numeric.

Notes
-----
- NaN values are preserved.
- If your feature is not naturally in (0, 1), rescale it first (e.g., min-max, sigmoid).
- The inverse transform is the logistic function: x = 1 / (1 + exp(-z)).
"""
```
📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-transformation/#logit)."


---
## **Negative Logarithm**

The `neg_log_transform` applies the **negative natural logarithm** to values in **(0, 1]**.  
It highlights small values (close to 0) by expanding them, while values closer to 1 remain compressed.  
This transformation is widely used in **p-value analysis** or whenever rare events need to be emphasized.

$$
f(x) = -\ln(x), \qquad x \in (0,1]
$$

**Typical use-cases**

- p-values (to emphasize significance)  
- Probabilities close to 0  
- Likelihood measures or bounded features requiring heavy emphasis near 0  

!!! tip "Numerical stability"
    Exact 0 is undefined. Quantreo **clips** to $[\,\varepsilon,\ 1\,]$ (default $\varepsilon=10^{-6}$).  
    The closer $x$ is to 0, the larger $-\log(x)$ becomes.

```python title="How to call the neg_log_transform function"
fe.transformations.non_linear.neg_log_transform(df: pd.DataFrame, col: str, eps: float = 1e-6)
```

```python title="neg_log_transform function docstring"
"""
Apply the negative logarithm transformation to a column in a DataFrame.

The transformation is defined as:
    f(x) = -log(x)

To avoid issues with log(0), values are clipped to [eps, 1].

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the data.
col : str
    Column name on which to apply the transformation. Values should lie in (0, 1].
eps : float, optional (default=1e-12)
    Small positive value used for clipping to prevent log(0).

Returns
-------
pd.Series
    Transformed series with the same index and name `{col}_neglog`.

Raises
------
ValueError
    If `col` does not exist or is not numeric.

Notes
-----
- NaN values are preserved.
- Use this transform when small values (near 0) carry more information.
- Typical for p-values or likelihood-based features.
"""
```
📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-transformation/#negative-logarithm)."


--- 
## **Causal Fourier Reconstruction**

The `fourier_transform` function applies a **Discrete Fourier Transform (DFT)** on a **rolling window** and reconstructs the signal while keeping only a subset of frequencies.  
At each step, it returns **the last reconstructed point** (causal), yielding a time-aligned feature you can use directly in backtests or live trading.

<br>

**Typical uses**:

- **Low-pass filtering** (`mode="lowpass"`) to smooth volatility or extract long-term trend  
- **Top-K dominant cycles** (`mode="topk"`) to capture the strongest periodic components in the market  
- Noise reduction by discarding high-frequency oscillations  

!!! warning "Stationarity"
    Since the Fourier basis is **global**, the choice of `window_size` is crucial.  
    Too small → poor frequency resolution.  
    Too large → risk of mixing non-stationary regimes.

!!! tip "Causal & NaNs"
    The output is **causal**: only past data inside the window are used.  
    The first `window_size − 1` values are **NaN** by construction.

```python title="How to call fourier_transform"
fe.transformations.signal.fourier_transform(df: pd.DataFrame, col: str, window_size: int = 256, mode: str = "topk", 
    top_k: int = 10, fmax_ratio: float | None = None, dt: float | None = None, keep_dc: bool = True,
    min_periods: int | None = None, name: str | None = None)
```

```
"""
Rolling Fourier reconstruction (pointwise) for feature engineering.

This function applies an FFT on a rolling window of length `window_size`,
keeps a restricted set of frequencies (Top-K or Low-pass), reconstructs
the window, and returns only the last reconstructed point at each step.

Parameters
----------
df : pd.DataFrame
    Input DataFrame containing the time series.
col : str
    Column name of the series to transform.
window_size : int, default=256
    Rolling window length (must be sufficient for frequency resolution).
mode : {"topk","lowpass"}, default="topk"
    - "topk"   : keep the K strongest positive frequencies (+ mirrors).
    - "lowpass": keep all frequencies below `fmax_ratio * Nyquist`.
top_k : int, default=10
    Number of dominant frequencies (used when mode="topk").
fmax_ratio : float or None, default=None
    Low-pass cutoff as a fraction of Nyquist (0<ratio<=1).
    Example: 0.2 keeps frequencies up to 20% of Nyquist.
dt : float or None, default=None
    Sampling step. If None and index is DatetimeIndex, inferred in seconds; else 1.0.
keep_dc : bool, default=True
    Keep the DC term (mean) in the reconstruction.
min_periods : int or None, default=None
    Minimum observations required in window. Defaults to `window_size` (causal).
name : str or None, default=None
    Name of the returned Series (auto-generated if None).

Returns
-------
pd.Series
    Rolling Fourier feature aligned with `df.index`. The first
    `min_periods-1` values are NaN.

Notes
-----
- 'topk' can jitter across windows if dominant bins change; prefer 'lowpass'
  for smoother, more stable features.
- This is causal when used with trailing windows (no look-ahead).
- If your sampling is irregular, resample before applying FFT-based transforms.

"""

```

📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-transformation/#causal-fourier-reconstruction)."

---
## **Causal Wavelet Reconstruction**

The `wavelet_transform` function applies a **Discrete Wavelet Transform (DWT)** on a **rolling window** and reconstructs the window while keeping only selected frequency bands.  
At each step it returns **the last reconstructed point** (causal), yielding a time-aligned feature you can use directly (e.g., smoothed trend or high-frequency component).

<br>

**Typical uses**:

- **Low-pass trend** (`keep="approx"`) for denoising and regime tracking  
- **High-frequency component** (`keep="details"`) to capture micro-microstructure or noise  
- **Full reconstruction** (`keep="all"`) as a reference or for residual analysis (original − reconstruction)

!!! warning "Window size vs. level"
    The decomposition **level** must be supported by the window length and the chosen wavelet.  
    This implementation **auto-clamps** the level to avoid boundary errors, but you should still pick a reasonable `window_size` (e.g., 128–512).

!!! tip "Causality & NaNs"
    The output is **causal**: only past data inside the window are used.  
    The first `window_size − 1` values are **NaN** by construction (insufficient history).

```python title="How to call wavelet_transform function"
fe.transformations.signal.wavelet_transform(series: pd.Series, window_size: int = 256, wavelet: str = "sym2",
    level: int = 3,  keep: str = "approx", mode: str = "symmetric", min_periods: int | None = None,
    name: str | None = None)
```

```python title="wavelet_transform function docstring"
"""
Rolling Wavelet reconstruction (pointwise) for feature engineering.

This function applies a DWT on a rolling window of length `window_size`,
reconstructs the window using the selected bands, and returns only the
last reconstructed point at each step. The result is a time-aligned Series
that can be used directly as a feature (e.g., low-pass trend or high-frequency
component).

Parameters
----------
series : pd.Series
    Input time series (NaNs are allowed; windows overlapping NaNs will return NaN).
window_size : int, default=256
    Rolling window length. Must be large enough to support the chosen `level`
    for the specified `wavelet`.
wavelet : str, default="sym2"
    Wavelet family (e.g., "haar", "db4", "sym5", "coif1").
level : int, default=3
    Decomposition level. Will be clamped to the maximum admissible level for
    `window_size` and `wavelet`.
keep : {"approx", "details", "all"}, default="approx"
    Which bands to keep in reconstruction:
      - "approx"  : keep only top-level approximation A_L (low-pass trend)
      - "details" : keep all details D_L..D1 (high-frequency content)
      - "all"     : full reconstruction (≈ original)
mode : str, default="symmetric"
    Signal extension mode used by PyWavelets.
min_periods : int or None, default=None
    Minimum observations in window required to have a value. If None,
    it defaults to `window_size` (strictly causal output).
name : str or None, default=None
    Name for the returned Series. If None, a descriptive name is generated.

Returns
-------
pd.Series
    Rolling wavelet feature aligned with `series.index`. The first
    `min_periods-1` values are NaN by design.

Raises
------
TypeError
    If `series` is not a pandas Series.
ValueError
    If `window_size` is too small for any wavelet decomposition with
    the chosen `wavelet`.

Notes
-----
- This is **causal** if you use a trailing window (`rolling(window_size)`).
- The function clamps the effective level for the chosen `window_size`, so you
  won't get PyWavelets boundary warnings.
- For exploration, consider producing both `"approx"` and `"details"` features.

Example
-------
>>> # Low-pass trend on closing prices (sym2, level 5, 256-window)
>>> df["ctc_close_wavelet"] = fe.transformations.signal.wavelet_transform(
...     series=df["close"],
...     window_size=256,
...     wavelet="sym2",
...     level=5,
...     keep="approx"
... )
"""

```
📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-transformation/#causal-wavelet-reconstruction)."


---
## **Median Moving Average**

The `mma` function computes the **Median Moving Average (MMA)** over a rolling window.  
Unlike the Simple Moving Average (SMA), which uses the mean, the MMA uses the **median**, making it **more robust to outliers and sudden price spikes**.


!!! tip "Robustness"
    If your data contains many **outliers or fat tails**, MMA is often preferred over SMA.  



```python title="How to call mma"
fe.features.trend.moving_averages.mma(df: pd.DataFrame,  col: str, window_size: int = 30)
```

```python title="mma function docstring"
"""
Calculate the Median Moving Average (MMA) using Pandas rolling.median.

The MMA smooths a time series by taking the median of values
over a fixed rolling window, making it more robust to outliers
compared to the Simple Moving Average (SMA).

Parameters
----------
df : pd.DataFrame
    DataFrame containing the input data.
col : str
    Column name on which to compute the MMA.
window_size : int, default=30
    The window size for computing the MMA (must be > 0).

Returns
-------
pd.Series
    A Series containing the MMA values, aligned with `df.index`.

Notes
-----
- The first (window_size - 1) values are NaN by design.
- More robust than SMA in presence of volatility spikes or price jumps.
- Useful for trend detection and noise reduction.
"""

```

📢 "For a practical example, check out the [educational notebook](/../tutorials/features-engineering-transformation/#median-moving-average)."
