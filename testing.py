import numpy as np
import pandas as pd
import time
from numba import njit


@njit(nogil=True)
def sma_numba(data, window):
    """
    Compute the Simple Moving Average (SMA) using a loop in pure NumPy, accelerated with Numba.

    Parameters
    ----------
    data : np.ndarray
        Array of numerical values.
    window : int
        The window size for computing the SMA.

    Returns
    -------
    sma : np.ndarray
        An array of SMA values (first window-1 elements are NaN).
    """
    n = data.shape[0]
    sma = np.empty(n)
    # Initialiser avec NaN
    for i in range(n):
        sma[i] = np.nan
    # Calcul de la SMA pour chaque fenêtre
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += data[j]
        sma[i] = s / window
    return sma


def sma_numpy(data, window):
    """
    Compute the Simple Moving Average (SMA) using a Python loop with NumPy functions.

    Parameters
    ----------
    data : np.ndarray
        Array of numerical values.
    window : int
        The window size for computing the SMA.

    Returns
    -------
    sma : np.ndarray
        An array of SMA values (first window-1 elements are NaN).
    """
    n = data.shape[0]
    sma = np.full(n, np.nan)
    for i in range(window - 1, n):
        sma[i] = np.sum(data[i - window + 1:i + 1]) / window
    return sma


# Création d'un DataFrame d'exemple avec 1 000 000 d'observations
df = pd.DataFrame({'values': np.random.rand(1_000_000)})
window = 50

# Mesure du temps avec Pandas rolling.mean
start = time.time()
sma_pandas = df['values'].rolling(window=window).mean().to_numpy()
time_pandas = time.time() - start

# Mesure du temps avec notre fonction Numba
start = time.time()
sma_nb = sma_numba(df['values'].to_numpy(), window)
time_numba = time.time() - start

# Mesure du temps avec la version pure Python utilisant NumPy (sans Numba)
start = time.time()
sma_py = sma_numpy(df['values'].to_numpy(), window)
time_pure = time.time() - start

print("Temps avec Pandas rolling.mean    :", time_pandas, "secondes")
print("Temps avec Numba (sma_numba)        :", time_numba, "secondes")
print("Temps avec boucle NumPy pure (sma_numpy):", time_pure, "secondes")
print("Erreur max entre Pandas et Numba   :", np.nanmax(np.abs(sma_pandas - sma_nb)))
print("Erreur max entre Pandas et boucle NumPy pure:", np.nanmax(np.abs(sma_pandas - sma_py)))
