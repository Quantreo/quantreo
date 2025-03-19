# **Market Regime**

You can find a series of examples on how to create these features in the [educational notebooks](/../tutorials/features-engineering-market-regime/) provided by Quantreo.

``` py
import quantreo.features_engineering as fe
```

---

## **KAMA Market Regime**

The `kama_market_regime` function generates a **market regime indicator** based on the relative position of two Kaufman's Adaptive Moving Averages (KAMA): a fast KAMA and a slow KAMA.

This function helps detect **bullish or bearish phases** by comparing a shorter-term KAMA with a longer-term KAMA:

- When the **fast KAMA > slow KAMA**, the market is in a **bullish regime** (`1`).
- When the **fast KAMA < slow KAMA**, the market is in a **bearish regime** (`-1`).

<br>

**Concept**
This method is inspired by the traditional **moving average crossover logic**, but with the adaptivity of KAMA smoothing.

The **fast KAMA** reacts more quickly to price changes due to its shorter parameters, while the **slow KAMA** filters out market noise over a longer period.

!!! tip Tip
    The choice of `l1`, `l2`, and `l3` parameters for each KAMA influences both **trend reactivity** and **noise filtering**. 
    For trend-following strategies, using `l1_fast = 50` and `l1_slow = 200` is common, but you can adjust them depending on your timeframe.

---

```python title="How to call the kama_market_regime function"
fe.market_regime.kama_market_regime(df: pd.DataFrame, col: str, 
                                    l1_fast=50, l2_fast=2, l3_fast=30, 
                                    l1_slow=200, l2_slow=2, l3_slow=30)
```

```
"""
Compute a market regime indicator based on the difference between two KAMA (fast and slow).

This function calculates two KAMA indicators using different parameter sets (fast and slow).
It then returns a regime signal:
- 1 if fast KAMA is above slow KAMA (bullish)
- -1 if fast KAMA is below slow KAMA (bearish)

Parameters
----------
df : pandas.DataFrame
    DataFrame containing the input price series.
col : str
    Column name on which to apply the KAMA calculation (e.g., 'close').
l1_fast : int, optional
    Efficiency ratio lookback window for the fast KAMA (default is 50).
l2_fast : int, optional
    Fastest EMA constant for the fast KAMA (default is 2).
l3_fast : int, optional
    Slowest EMA constant for the fast KAMA (default is 30).
l1_slow : int, optional
    Efficiency ratio lookback window for the slow KAMA (default is 200).
l2_slow : int, optional
    Fastest EMA constant for the slow KAMA (default is 2).
l3_slow : int, optional
    Slowest EMA constant for the slow KAMA (default is 30).

Returns
-------
pandas.Series
    A Series containing the market regime indicator:
    - 1 for bullish regime
    - -1 for bearish regime
"""

```
📢 *For a practical example, check out this [educational notebook](/../tutorials/features-engineering-market-regime/#kama-market-regime).*
