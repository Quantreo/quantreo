import numpy as np
import pandas as pd
import antropy as ant


def sample_entropy(df: pd.DataFrame, col: str = "close", window_size: int = 100, order: int = 2) -> pd.Series:
    """
    Calculate the rolling Sample Entropy of a time series.

    Sample Entropy quantifies the level of irregularity or unpredictability
    in a signal. It is often used to detect dynamic changes in structural
    complexity over time.

    This function applies Sample Entropy on a sliding window, allowing you
    to observe how entropy evolves across a time series.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the time series.
    col : str, default="close"
        The name of the column on which to compute the entropy.
    window_size : int, default=100
        Size of the rolling window (must be >= 10).
    order : int, default=2
        Embedding dimension used in the entropy calculation (must be >= 1).

    Returns
    -------
    pd.Series
        A Series containing the rolling Sample Entropy values. The first
        (window_size - 1) values will be NaN.

    Notes
    -----
    This function uses AntroPy's implementation of Sample Entropy.
    AntroPy is licensed under the BSD 3-Clause License.
    © 2018–2025 Raphael Vallat — https://github.com/raphaelvallat/antropy
    """
    if window_size < 10:
        raise ValueError("Sample entropy requires window_size >= 10.")

    if order < 1:
        raise ValueError("Parameter 'order' must be >= 1.")

    return df[col].rolling(window=window_size).apply(
        lambda x: ant.sample_entropy(x, order=order) if len(x) == window_size else np.nan,
        raw=True)


def spectral_entropy(df: pd.DataFrame, col: str = "close", window_size: int = 100, sf: int = 1,
                             method: str = 'welch', normalize: bool = True, nperseg: int = None) -> pd.Series:
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

    Notes
    -----
    This function uses AntroPy's implementation of Spectral Entropy.
    AntroPy is licensed under the BSD 3-Clause License.
    © 2018–2025 Raphael Vallat — https://github.com/raphaelvallat/antropy
    """
    if window_size < 16:
        raise ValueError("Spectral entropy requires window_size >= 16 for stable estimation.")

    if sf <= 0:
        raise ValueError("Sampling frequency 'sf' must be strictly positive.")

    if method not in ["welch", "fft"]:
        raise ValueError("Method must be 'welch' or 'fft'.")

    def compute_entropy(x):
        local_nperseg = nperseg if nperseg is not None else min(window_size, window_size // 2)
        return ant.spectral_entropy(x, sf=sf, method=method, normalize=normalize, nperseg=local_nperseg)

    return df[col].rolling(window=window_size).apply(
        lambda x: compute_entropy(x) if len(x) == window_size else np.nan,
        raw=True)


def permutation_entropy(df: pd.DataFrame, col: str = "close", window_size: int = 100, order: int = 3,
                                delay: int = 1, normalize: bool = True) -> pd.Series:
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

    Notes
    -----
    This function uses AntroPy's implementation of Permutation Entropy.
    AntroPy is licensed under the BSD 3-Clause License.
    © 2018–2025 Raphael Vallat — https://github.com/raphaelvallat/antropy
    """
    if window_size < 10:
        raise ValueError("Permutation entropy requires window_size >= 10.")

    if order < 2:
        raise ValueError("Embedding 'order' must be >= 2.")

    if delay < 1:
        raise ValueError("Delay must be >= 1.")

    return df[col].rolling(window=window_size).apply(
        lambda x: ant.perm_entropy(x, order=order, delay=delay, normalize=normalize)
        if len(x) == window_size else np.nan,
        raw=True)


def petrosian_fd(df: pd.DataFrame, col: str = "close", window_size: int = 100) -> pd.Series:
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

    Notes
    -----
    This function uses AntroPy's implementation of Petrosian FD.
    AntroPy is licensed under the BSD 3-Clause License.
    © 2018–2025 Raphael Vallat — https://github.com/raphaelvallat/antropy
    """
    if window_size < 10:
        raise ValueError("Petrosian fractal dimension requires window_size >= 10.")

    return df[col].rolling(window=window_size).apply(
        lambda x: ant.petrosian_fd(x) if len(x) == window_size else np.nan,
        raw=True)