# ✨ **Getting Started with Quantreo**

##  **Introduction to Quantreo**

**Welcome to the beginner's guide to Quantreo!**  

Quantreo is a Python library designed for **algorithmic trading** and **quantitative finance**. For now, it simplifies **Feature Engineering**, and soon, much more packages will be added.  

<br> 

💡 **Why use Quantreo?**  
✔ Standardizes and automates common data preprocessing in finance  
✔ Compatible with **scikit-learn, pandas, NumPy**  
✔ Optimized with **Numba** and **vectorized calculations** for high performance  

!!! info "Coming Soon"
    Target Engineering will be added soon!

<br>

---

## **Installation and Import**

### **Installation**
Install Quantreo with **pip**:

```bash
pip install quantreo
```

### **Import**
After installation, import the Quantreo's packages like this:

```python
# Import The Features Engineering Package
import quantreo.features_engineering as fe

# Import The Target Engineering Package
import quantreo.target_engineering as te
```
<br>


---
## **Data Format**

Quantreo is designed to work with **OHLCV (Open, High, Low, Close, Volume) data**, which is the standard format in financial markets.  


!!! success "🔗 OHLCV focus"
    <u>**Not all features require all columns**</u>. So, if you compute a features needing only the high and the low prices, you can put a dataframe containing only these two variables.
     

Let's create really a very simple dataframe to show you the right format.
```python
import pandas as pd

# Creating a sample OHLCV DataFrame
data = {
    "open": [100, 102, 101, 103, 105],
    "high": [105, 107, 106, 108, 110],
    "low": [98, 100, 99, 101, 102],
    "close": [102, 104, 103, 105, 107],
    "volume": [10000, 10050, 9950, 10000, 11500]
}
df = pd.DataFrame(data)

print(df)
```

| Open | High | Low | Close | Volume |
|------|------|-----|-------|--------|
| 100  | 105  |  98 |  102  | 10000  |
| 102  | 107  | 100 |  104  | 10050  |
| 101  | 106  |  99 |  103  | 9950   |
| 103  | 108  | 101 |  105  | 10000  |
| 105  | 110  | 102 |  107  | 11500  |

<br>

---

## **The Feature Engineering Package**

### **What is Feature Engineering?**
In **quantitative trading**, Feature Engineering transforms **raw market data** into indicators that can be used in a model.

Example: Calculating **volatility, trend, logarithmic variation**, etc.  
<br>

With Quantreo, extracting this information is **optimized and efficient**.


Here’s how to use a **Feature Engineering function** in Quantreo. Let's calculate **Yang-Zhang Volatility** (using our previous dataframe):

```python
import quantreo.features_engineering as fe

# Compute Yang-Zhang Volatility with a 3-period rolling window
df["yang_zhang_vol"] = fe.volatility.yang_zhang_volatility(df, window_size=3)

# Compute the spread between the high and low price
df["spread"] = fe.candle.compute_spread(df)

print(df)
```

| Open | High | Low | Close | Volume | YZ Vol   | Spread |
|------|------|-----|-------|--------|----------|--------|
| 100  | 105  |  98 |  102  | 10000  | Nan      | 7      |
| 102  | 107  | 100 |  104  | 10050  | Nan      | 7      |
| 101  | 106  |  99 |  103  | 9950   | Nan      | 7      |
| 103  | 108  | 101 |  105  | 10000  | 0.044204 | 7      |
| 105  | 110  | 102 |  107  | 11500  | 0.043777 | 8      |


!!! warning "Nan values"
    It is 100% normal to have 3 Nan values into the first rows of the `YZ vol` column. It is because we compute the **volatily over the 3 last observations**. So, it is impossible to compute these values as we do not have the necessary information

!!! info "The Features Structure"
    To call a feature, we always use the same logic. If I have imported the features engineering package of Quantreo as fe, I will write
    `fe.features_category.features_fonction(param1=..., param2=...)`.



<br>

---

## **Coming Soon**

Quantreo will soon include **Target Engineering** functionalities.

✔ Define **robust target variables** for supervised learning  
✔ Adapt target definitions to **various trading strategies**  
✔ Handle **prediction horizons** and complex **labeling** mechanisms  

📢 *Stay tuned for the next updates!*

---

## **Learn More**

📚 Take a look to our "Machine Learning for Trading" course !
