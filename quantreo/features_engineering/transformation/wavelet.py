import numpy as np
import pandas as pd
import pywt


def _wavelet_last_point(x: np.ndarray, wavelet: str = "sym2", level: int = 3, keep: str = "approx",
    mode: str = "symmetric") -> float:
    """
    Compute the last reconstructed value of a 1D window using a DWT.

    This helper is designed to be called inside a rolling window. It:
      1) performs a Discrete Wavelet Transform (DWT) on the window,
      2) keeps only the requested bands (approx/details/all),
      3) reconstructs the signal,
      4) returns the **last** reconstructed sample.

    Parameters
    ----------
    x : np.ndarray
        1D window (latest sample is x[-1]).
    wavelet : str, default="db4"
        Wavelet family (e.g., "haar", "db4", "sym5", "coif1").
    level : int, default=3
        Decomposition level. The function automatically clamps it to the
        maximum admissible level for `len(x)` and the chosen wavelet.
    keep : {"approx", "details", "all"}, default="approx"
        Which bands to keep for reconstruction:
          - "approx"  : keep only the top-level approximation A_L (low-pass)
          - "details" : keep all details D_L..D1 (high-pass)
          - "all"     : full reconstruction (≈ original window)
    mode : str, default="symmetric"
        Signal extension mode used by PyWavelets ("symmetric", "periodization", ...).

    Returns
    -------
    float
        The last value of the reconstructed window.

    Notes
    -----
    - This function is **stateless** and fast enough for rolling.apply.
    - Level is clamped with `pywt.dwt_max_level` to avoid boundary warnings.
    """
    # Ensure float computations
    x = np.asarray(x, dtype=float)

    # Clamp decomposition level to what the window supports
    w = pywt.Wavelet(wavelet)
    Lmax = pywt.dwt_max_level(len(x), w.dec_len)
    if Lmax < 1:
        # Not enough points for any decomposition with this wavelet/window
        return float(np.nan)
    L = int(min(max(1, level), Lmax))

    # DWT
    coeffs = pywt.wavedec(x, wavelet=w, level=L, mode=mode)

    # Keep only requested bands
    coeffs_mod = [np.zeros_like(c) for c in coeffs]
    keep_key = str(keep).lower()
    if keep_key == "approx":
        coeffs_mod[0] = coeffs[0]
    elif keep_key == "details":
        coeffs_mod[1:] = coeffs[1:]
    elif keep_key == "all":
        coeffs_mod = coeffs
    else:
        raise ValueError("Invalid `keep`. Use 'approx', 'details', or 'all'.")

    # Reconstruct and return the last sample
    recon = pywt.waverec(coeffs_mod, wavelet=w, mode=mode)
    return float(recon[-1])


def wavelet_transform(
    df: pd.DataFrame,
    col: str,
    window_size: int = 256,
    wavelet: str = "db4",
    level: int = 3,
    keep: str = "approx",
    mode: str = "symmetric",
    min_periods: int | None = None,
    name: str | None = None,
) -> pd.Series:
    """
    Rolling Wavelet reconstruction (pointwise) for feature engineering.

    This function applies a DWT on a **rolling window** of length `window_size`,
    reconstructs the window using the selected bands, and returns **only the
    last reconstructed point** at each step. The result is a time-aligned series
    that can be used directly as a feature (e.g., low-pass trend or high-frequency
    component).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series.
    col : str
        Column name of the series to transform.
    window_size : int, default=256
        Rolling window length. Must be large enough to support the chosen `level`
        for the specified `wavelet`.
    wavelet : str, default="db4"
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
        Rolling wavelet feature aligned with `df.index`. The first
        `min_periods-1` values are NaN by design.

    Raises
    ------
    ValueError
        If `col` is not in DataFrame or if `window_size` is too small for the wavelet.

    Notes
    -----
    - This is **causal** if you use a trailing window (`rolling(window_size)`).
    - The function clamps the effective level for the chosen `window_size`, so you
      won't get PyWavelets boundary warnings.
    - For EDA, consider producing both `"approx"` and `"details"` features.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    series = df[col]

    # Validate that the window can support at least level=1 with this wavelet
    w = pywt.Wavelet(wavelet)
    Lmax_possible = pywt.dwt_max_level(window_size, w.dec_len)
    if Lmax_possible < 1:
        raise ValueError(
            f"`window_size={window_size}` is too small for wavelet '{wavelet}'. "
            f"Increase the window or choose a shorter wavelet."
        )

    # Clamp requested level to what the window supports
    level_eff = int(min(max(1, level), Lmax_possible))

    # Build a fast partial function for rolling.apply (raw=True passes a numpy array)
    def _apply(arr: np.ndarray) -> float:
        return _wavelet_last_point(arr, wavelet=wavelet, level=level_eff, keep=keep, mode=mode)

    mp = window_size if min_periods is None else min_periods
    out = series.rolling(window=window_size, min_periods=mp).apply(_apply, raw=True)

    if name is None:
        name = f"{col}_wavelet_{keep}_L{level_eff}_{wavelet}"
    return out.rename(name)

