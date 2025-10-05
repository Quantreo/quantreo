# **Volatility**
You can find a series of examples on how to create these features in the [educational notebooks](/../tutorials/features-engineering-volatility) provided by Quantreo.



``` py
import quantreo.features_engineering as fe
```

!!! warning "Warning"
    Volatility values depend on both the price scale and the timeframe you are analyzing. Higher timeframes or assets with larger price magnitudes can exhibit higher absolute volatility. Always normalize volatility when comparing across different assets or timeframes.


---
## **CTC Volatility**

The `close_to_close_volatility` function calculates the **close-to-close volatility**, a widely used measure of market risk based on the standard deviation of **log returns**. This approach captures **price fluctuations over time** and is particularly useful in **statistical modeling and risk management**. The **close-to-close volatility** is defined as:

\[
\sigma_{CC} = \sqrt{\frac{1}{N - 1} \sum_{i=1}^{N} \left( \ln \frac{C_i}{C_{i-1}} - \mu \right)^2}
\]

Where:

- **\(C_i\)**: The **closing price** at time \(i\).
- **\(N\)**: The number of periods in the rolling window.
- **\(\mu\)**: The mean of the log returns over the window.


=== "Function"
    ```python
    fe.volatility.close_to_close_volatility(df: pd.DataFrame, window_size: int = 30, close_col: str = 'close')
    ```

=== "Docstring"
    ```python
    """
    Calculate the rolling close-to-close volatility.md using standard deviation.
    
    This method computes the rolling standard deviation of the log returns,
    which represents the close-to-close volatility.md over a specified window.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the price data.
    window_size : int, optional
        The number of periods to include in the rolling calculation (default is 30).
    close_col : str, optional
        Column name for the closing prices (default is 'close').
    
    Returns
    -------
    volatility_series : pd.Series
        A Series indexed the same as `df`, containing the rolling close-to-close volatility.md.
        The first `window_size` rows will be NaN because there is insufficient data
        to compute the volatility.md in those windows.
    """
    ```

=== "Example"
    ```python
    df["ctc_vol"] = fe.volatility.close_to_close_volatility(df=df, close_col="close", window_size=30)
    ```

📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-volatility/#ctc-volatility).*

---
## **Parkinson Volatility**

The `parkinson_volatility` function computes the **Parkinson volatility**, an estimator designed to measure market volatility using only **high and low prices**. This approach provides a more accurate estimate than simple **close-to-close volatility**, particularly in **markets with low closing price variation but significant intraday movement**.

!!! tip "Tip"
    Unlike standard volatility measures, the **Parkinson estimator** is a **rolling volatility measure**, meaning that at each time step, the function calculates the volatility over the **last N observations**, producing a **time series of volatility values**.

The **Parkinson estimator** is defined as:

\[
\sigma_P^2 = \frac{1}{4N \ln(2)} \sum_{i=1}^{N} \left( \ln \frac{h_i}{l_i} \right)^2
\]

*(Source: Parkinson, 1980, "The Extreme Value Method for Estimating the Variance of the Rate of Return")*

Where:

- **\(h_i\)**: The **high price** at time \(i\).
- **\(l_i\)**: The **low price** at time \(i\).
- **\(N\)**: The number of periods in the rolling window.

!!! important "Important"
    This estimator is **more accurate than close-to-close volatility** because it:

    ✔ Uses **high and low prices**, which contain more information than just closing prices.  
    ✔ Provides a **rolling measure of volatility**, making it suitable for **time-series analysis**.  
    ✔ Assumes that price movements follow a **Brownian motion** without drift.  

=== "Function"
    ```python
    fe.volatility.parkinson_volatility(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low', window_size: int = 30)
    ``` 

=== "Docstring"
    ``` python
    """
    Calculate Parkinson's volatility.md estimator using numpy operations with Numba acceleration.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the price data.
    high_col : str, optional
        Column name for the high prices (default is 'high').
    low_col : str, optional
        Column name for the low prices (default is 'low').
    window_size : int, optional
        The number of periods to include in the rolling calculation (default is 30).
    
    Returns
    -------
    volatility_series : pandas.Series
        A Series indexed the same as `df`, containing the rolling Parkinson volatility.md.
        The first `window_size` rows will be NaN because there is insufficient data
        to compute the volatility.md in those windows.
    """
    ```

=== "Example"
    ```python
    df["parkinson_vol"] = fe.volatility.parkinson_volatility(df=df, high_col="high", low_col="low", window_size=30)
    ```

📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-volatility/#parkinson-volatility).*




---
## **Rogers-Satchell Volatility**
The `rogers_satchell_volatility` function computes the **Rogers-Satchell volatility**, an estimator designed specifically for **assets with a directional drift**, making it more suitable for financial markets than standard measures like **close-to-close volatility**.

!!! tip "Tip"
    Unlike standard volatility measures, the **Rogers-Satchell estimator** is a **rolling volatility** measure, meaning that at each time step, the function calculates the volatility over the **last N observations**, producing a **time series of volatility values**.

The **Rogers-Satchell estimator** is defined as:

\[
\sigma_{RS}^2 = \frac{1}{N} \sum_{i=1}^{N} \left( \ln (\frac{h_i}{c_i}) \ln (\frac{h_i}{o_i}) + \ln (\frac{l_i}{c_i}) \ln (\frac{l_i}{o_i}) \right)
\]

*(Source: Rogers & Satchell, 1994, "Estimating Variance from High, Low, and Close Prices")*

Where:

- **\(h_i\)**: The **high price** at time \(i\).
- **\(l_i\)**: The **low price** at time \(i\).
- **\(o_i\)**: The **open price** at time \(i\).
- **\(c_i\)**: The **close price** at time \(i\).
- **\(N\)**: The number of periods in the rolling window.

!!! important "Important"
    This estimator is **more accurate than standard close-to-close volatility** because it:

    ✔ Incorporates **intraday high and low prices**, providing a better estimate of actual price movement.  
    ✔ Does not assume zero drift, making it more applicable to **trending markets**.  
    ✔ Is **computationally efficient** while reducing bias in the variance estimate.  

=== "Function"
    ```python
    fe.volatility.rogers_satchell_volatility(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low', open_col: str = 'open',
                                   close_col: str = 'close', window_size: int = 30)
    ```

=== "Docstring"
    ```python
    """
    Compute the Rogers-Satchell volatility estimator using a rolling window.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing OHLC price data.
    high_col : str, optional
        Column name for high prices (default = 'high').
    low_col : str, optional
        Column name for low prices (default = 'low').
    open_col : str, optional
        Column name for open prices (default = 'open').
    close_col : str, optional
        Column name for close prices (default = 'close').
    window_size : int, optional
        The number of periods used in the rolling calculation (default = 30).
    
    Returns
    -------
    pd.Series
        A Series containing the rolling Rogers-Satchell volatility, indexed like `df`.
        The first `window_size` rows are NaN due to insufficient data.
    """
    ```

=== "Example"
    ```python
    df["rs_vol"] = fe.volatility.rogers_satchell_volatility(df=df, high_col="high", low_col="low",open_col="open",
                                                            close_col="close", window_size=30)
    ```

📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-volatility/#rogers-satchell-volatility).*


---

## **Yang-Zhang Volatility**

The `yang_zhang_volatility` function computes the **Yang-Zhang volatility**, an advanced estimator that integrates **overnight, close-to-close, and intra-day price movements**. This method is particularly useful for assets with **significant overnight gaps**, making it more robust than traditional measures.

!!! tip "Tip"
    Unlike standard volatility measures, the **Yang-Zhang estimator** is a **rolling volatility measure**, meaning that at each time step, the function calculates the volatility over the **last N observations**, producing a **time series of volatility values**.


The **Yang-Zhang estimator** is defined as:

\[
\sigma_t = \sqrt{\sigma_O^2 + k\sigma_C^2 + (1 - k) \sigma_{RS}^2}
\]

*(Source: Yang & Zhang, 2000, "Drift Independent Volatility Estimation")*

Where:

- **\(\sigma_O^2\)**: The **overnight volatility**, measuring the variance between **close-to-open** prices.
- **\(\sigma_C^2\)**: The **close-to-close volatility**, capturing daily price changes.
- **\(\sigma_{RS}^2\)**: The **Rogers-Satchell estimator**, which accounts for intra-day price movements.
- **\(k\)**: A weighting factor empirically determined by Yang & Zhang, according to them, the best empirical value is 0.34.

!!! important "Important"
    This estimator is **more accurate than simple close-to-close volatility** because it:

    ✔ Reduces bias from opening gaps.  
    ✔ Accounts for intra-day price fluctuations.  
    ✔ Provides a better measure of realized volatility.

=== "Function"
    ```python
    fe.volatility.yang_zhang_volatility(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low',
                              open_col: str = 'open', close_col: str = 'close', window_size: int = 30, k: float = 0.34)
    ```

=== "Docstring"
    ```python
        """
        Compute the Yang-Zhang volatility estimator using a rolling window.
    
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing OHLC price data.
        high_col : str, optional
            Column name for high prices (default = 'high').
        low_col : str, optional
            Column name for low prices (default = 'low').
        open_col : str, optional
            Column name for open prices (default = 'open').
        close_col : str, optional
            Column name for close prices (default = 'close').
        window_size : int, optional
            The number of periods used in the rolling calculation (default = 30).
        k : float, optional
            The weighting parameter for the open-to-close variance component, as described
            in Yang & Zhang (2000). Empirical research suggests 0.34 as the optimal value.
    
        Returns
        -------
        pd.Series
            A Series containing the rolling Yang-Zhang volatility, indexed like `df`.
            The first `window_size` rows are NaN due to insufficient data.
        """
    ```

=== "Example"
    ```python
    df["yz_vol"] = fe.volatility.yang_zhang_volatility(df=df, low_col="low",high_col="high", open_col="open",
                                                       close_col="close", window_size=30)
    ```

📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-volatility/#yang-zhang-volatility-estimator).*

