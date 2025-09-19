import numpy as np
import pandas as pd


def _fourier_last_point(x: np.ndarray, mode: str = "topk", top_k: int = 10, fmax_ratio: float | None = None,
    dt: float = 1.0, keep_dc: bool = True) -> float:
    """
    Return the last reconstructed value of a window using a band-limited FFT.

    Strategy
    --------
    - Compute FFT on the window.
    - Keep either:
        (mode="topk") the K strongest positive frequencies (+ their mirrors),
        OR
        (mode="lowpass") all frequencies below fmax_ratio * Nyquist.
    - Optionally keep DC (mean).
    - IFFT and return the **last** sample of the reconstructed window.

    Notes
    -----
    - Each window is processed independently (causal rolling usage).
    - 'topk' may jitter when the dominant bins change; 'lowpass' is stabler for features.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if N < 4:
        return float("nan")

    F = np.fft.fft(x)

    if mode == "topk":
        # Start from an empty spectrum
        Fk = np.zeros_like(F, dtype=complex)
        if keep_dc:
            Fk[0] = F[0]

        half = N // 2
        if N % 2 == 0:
            # even length: positive bins are 1..(half-1), Nyquist is at half (self-conjugate)
            pos_bins = np.arange(1, half)
            nyq_idx = half
        else:
            # odd length: positive bins are 1..half
            pos_bins = np.arange(1, half + 1)
            nyq_idx = None

        amps = np.abs(F[pos_bins])
        k = int(min(top_k, amps.size))
        if k > 0:
            sel_pos = pos_bins[np.argpartition(amps, -k)[-k:]]
            # Keep ±f pairs to maintain real-valued reconstruction
            Fk[sel_pos] = F[sel_pos]
            Fk[-sel_pos] = F[-sel_pos]

        # Optionally keep Nyquist (rarely makes a difference)
        if nyq_idx is not None:
            # Keep it only if it is among the strongest bins beyond 'top_k' heuristic?
            # Here we ignore it for stability; uncomment next line to always keep it.
            # Fk[nyq_idx] = F[nyq_idx]
            pass

    elif mode == "lowpass":
        if fmax_ratio is None:
            raise ValueError("`fmax_ratio` is required when mode='lowpass'.")
        freqs = np.fft.fftfreq(N, d=dt)
        nyq = 0.5 / dt
        mask = np.abs(freqs) <= (float(fmax_ratio) * nyq)
        Fk = np.where(mask, F, 0.0 + 0.0j)
        if not keep_dc:
            Fk[0] = 0.0
    else:
        raise ValueError("`mode` must be 'topk' or 'lowpass'.")

    recon_last = np.fft.ifft(Fk).real[-1]
    return float(recon_last)


def fourier_transform(df: pd.DataFrame, col: str, window_size: int = 256,  mode: str = "topk",
    top_k: int = 10,  fmax_ratio: float | None = None,  dt: float | None = None,  keep_dc: bool = True,
    min_periods: int | None = None,  name: str | None = None) -> pd.Series:
    """
    Rolling Fourier reconstruction (pointwise) for feature engineering.

    This function applies an FFT on a **rolling window** of length `window_size`,
    keeps a restricted set of frequencies (Top-K or Low-pass), reconstructs the
    window, and returns **only the last reconstructed point** at each step.
    The result is a time-aligned feature usable in backtests and live.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series.
    col : str
        Column name of the series to transform.
    window_size : int, default=256
        Rolling window length (must be sufficient for your frequency resolution).
    mode : {"topk","lowpass"}, default="topk"
        - "topk"   : keep the K strongest positive frequencies (+ mirrors).
        - "lowpass": keep all frequencies below `fmax_ratio * Nyquist`.
    top_k : int, default=10
        Number of dominant frequencies (used when mode="topk").
    fmax_ratio : float or None, default=None
        Low-pass cutoff **as a fraction of Nyquist** (0<ratio<=1), used when mode="lowpass".
        Example: 0.2 keeps frequencies up to 20% of Nyquist.
    dt : float or None, default=None
        Sampling step. If None and index is a DatetimeIndex, inferred in seconds; else 1.0.
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
    - **'topk' can jitter** across windows if the strongest bins change; prefer 'lowpass'
      for smoother, more stable features.
    - This is **causal** when used with trailing windows (no look-ahead).
    - If your sampling is irregular, resample before using FFT-based transforms.

    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    series = df[col]

    # infer dt if possible
    if dt is None:
        if isinstance(series.index, pd.DatetimeIndex) and len(series.index) > 1:
            step = pd.Series(series.index).diff().dt.total_seconds().median()
            dt_eff = float(step) if pd.notna(step) and step > 0 else 1.0
        else:
            dt_eff = 1.0
    else:
        dt_eff = float(dt)

    mp = window_size if min_periods is None else int(min_periods)

    def _apply(arr: np.ndarray) -> float:
        return _fourier_last_point(
            arr,
            mode=mode,
            top_k=top_k,
            fmax_ratio=fmax_ratio,
            dt=dt_eff,
            keep_dc=keep_dc,
        )

    out = series.rolling(window=window_size, min_periods=mp).apply(_apply, raw=True)
    if name is None:
        suffix = f"{mode}_k{top_k}" if mode == "topk" else f"lp_{fmax_ratio}"
        name = f"{col}_fft_{suffix}"
    return out.rename(name)
