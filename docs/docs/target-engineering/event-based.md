# **Event-Based Targets**
You can find several examples on how to create these targets in the [educational notebooks](/../tutorials/target-engineering-event-based) provided by Quantreo.

```python
import quantreo.target_engineering as te
```

---

## **Peaks & Valleys Detection**

The `detect_peaks_valleys` function identifies **structural turning points** in a price series by labeling **local maxima (peaks)** and **local minima (valleys)**. It is particularly useful to define event-driven targets that are independent of fixed time intervals.

- The function returns a label of `1` for a **peak**, `-1` for a **valley**, and `0` elsewhere.
- Internally, it leverages `scipy.signal.find_peaks` for robustness.

!!! tip "Tip"
    You can combine this function with magnitude-based targets like `future_returns()` to detect **how strong the move is after a turning point**.

```python title="How to call detect_peaks_valleys"
te.event_based.detect_peaks_valleys(df: pd.DataFrame, col: str = 'close', distance: int = 5, prominence: float = 0.5)
```

``` title="detect_peaks_valleys docstring"
"""
Detect peaks and valleys in a time series using scipy's find_peaks.

This function labels turning points in a price series:
- 1 for local maxima (**peaks**),
- -1 for local minima (**valleys**),
- 0 for all other points.

It internally uses `scipy.signal.find_peaks` for both peak and valley detection.

Parameters
----------
df : pd.DataFrame
    DataFrame containing the price data.
col : str, optional
    The column name of the series to analyze (default is 'close').
distance : int, optional
    Minimum number of samples between two peaks or valleys.
    This parameter is passed to `scipy.signal.find_peaks`.
    A higher value will ignore smaller fluctuations and detect only well-separated turning points.
prominence : float, optional
    Required prominence of peaks/valleys.
    Also passed to `scipy.signal.find_peaks`.
    Prominence represents how much a peak stands out from the surrounding data (in the "col" input unit).
    A higher value filters out smaller movements and keeps only significant peaks/valleys.

Returns
-------
pd.Series
    A Series of labels with the same index as `df`:
    - 1 for peaks,
    - -1 for valleys,
    - 0 for neutral points.
"""

```

📢 "For a practical example, check out the [educational notebook](/../tutorials/target-engineering-event-based/#peaks-valleys-detection)."

