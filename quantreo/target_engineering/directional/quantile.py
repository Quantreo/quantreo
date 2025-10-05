import pandas as pd


def quantile_label(
    df: pd.DataFrame,
    col: str,
    upper_quantile_level: float = 0.67,
    lower_quantile_level: float | None = None,
    q_high: float | None = None,
    q_low: float | None = None,
    return_thresholds: bool = False,
    positive_label: int = 1,
    negative_label: int = -1,
    neutral_label: int = 0,
) -> pd.Series | tuple[pd.Series, float, float]:
    """
    Generate quantile-based labels (custom values for upper, lower, and neutral) and optionally return thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the target column (e.g., 'fut_ret').
    col : str
        Name of the column used for quantiles (e.g., 'fut_ret').
    upper_quantile_level : float, optional (default=0.67)
        The quantile level for the upper threshold.
    lower_quantile_level : float or None, optional (default=None)
        The quantile level for the lower threshold.
        If None, defaults to `1 - upper_quantile_level`.
    q_high : float or None, optional (default=None)
        Pre-calculated upper quantile value.
    q_low : float or None, optional (default=None)
        Pre-calculated lower quantile value.
    return_thresholds : bool, optional (default=False)
        If True, returns both the labels and the thresholds.
    positive_label : int or any, optional (default=1)
        Value assigned when the value is above the upper quantile.
    negative_label : int or any, optional (default=-1)
        Value assigned when the value is below the lower quantile.
    neutral_label : int or any, optional (default=0)
        Value assigned when the value is between the two quantiles.

    Returns
    -------
    labels : pandas.Series
        Series of custom labels.
    q_high : float (optional)
        Upper quantile value (if return_thresholds is True).
    q_low : float (optional)
        Lower quantile value (if return_thresholds is True).
    """

    if lower_quantile_level is None:
        lower_quantile_level = 1 - upper_quantile_level

    if q_high is None:
        q_high = df[col].quantile(upper_quantile_level)
    if q_low is None:
        q_low = df[col].quantile(lower_quantile_level)

    if q_high <= q_low:
        raise ValueError("Invalid quantiles: q_high must be strictly greater than q_low.")

    labels = pd.Series(neutral_label, index=df.index)
    labels.loc[df[col] > q_high] = positive_label
    labels.loc[df[col] < q_low] = negative_label

    if return_thresholds:
        return labels, q_high, q_low
    else:
        return labels
