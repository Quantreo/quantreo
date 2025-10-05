from numba import njit
import numpy as np


@njit
def skewness(x: np.ndarray) -> float:
    """
    Compute the skewness of a 1D array.

    Skewness measures the asymmetry of the distribution:
    - A value > 0 indicates right-skewed data (long tail on the right)
    - A value < 0 indicates left-skewed data (long tail on the left)
    - A value close to 0 suggests symmetry

    Parameters
    ----------
    x : np.ndarray
        Input 1D array of numerical values (e.g., prices or volumes within a bar).

    Returns
    -------
    float
        Skewness value of the input array.
    """
    n = len(x)
    if n < 2:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return np.mean(((x - mean) / std) ** 3)


@njit
def kurtosis(x: np.ndarray) -> float:
    """
    Compute the excess kurtosis of a 1D array using the Fisher definition.

    Kurtosis measures the "tailedness" of a distribution:
    - Normal distribution returns ~0
    - High kurtosis indicates more extreme outliers (heavy tails)
    - Low kurtosis indicates fewer and less extreme outliers

    Parameters
    ----------
    x : np.ndarray
        Input 1D array of numerical values (e.g., prices or volumes within a bar).

    Returns
    -------
    float
        Excess kurtosis value. 0 for Gaussian-like distribution.
    """
    n = len(x)
    if n < 4:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    m4 = np.mean(((x - mean) / std) ** 4)
    return m4 - 3.0  # Fisher's definition of excess kurtosis
