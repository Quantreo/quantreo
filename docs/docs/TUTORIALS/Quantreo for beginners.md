# ✨ **Getting Started with Quantreo**

##  **Introduction to Quantreo**

**Welcome to the beginner's guide to Quantreo!**  

Quantreo is a Python library designed for **algorithmic trading** and **quantitative finance**. For now, it simplifies **Feature Engineering**, and soon, much more packages will be added.  


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
```
**DataFrame Obtained**:

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

✅ Example: Calculating **volatility, trend, logarithmic variation**, etc.  

With Quantreo, extracting this information is **optimized and efficient**.


Here’s how to use a **Feature Engineering function** in Quantreo. Let's calculate **Yang-Zhang Volatility** (using our previous dataframe):

```python
import quantreo.features_engineering as fe

# Compute Yang-Zhang Volatility with a 3-period rolling window
df["yang_zhang_vol"] = fe.yang_zhang_volatility(df, window_size=3)

print(df)
```

| Open | High | Low | Close | Volume | YZ Vol |
|------|------|-----|-------|--------|--------|
| 100  | 105  |  98 |  102  | 10000  | Nan    |
| 102  | 107  | 100 |  104  | 10050  | Nan    |
| 101  | 106  |  99 |  103  | 9950   | Nan    | 
| 103  | 108  | 101 |  105  | 10000  |
| 105  | 110  | 102 |  107  | 11500  |










## 📊 Understanding Feature Engineering

### **What is Feature Engineering?**
In **quantitative trading**, Feature Engineering transforms **raw market data** into indicators that can be used in a model.

✅ Example: Calculating **volatility, trend, logarithmic variation**, etc.  

With Quantreo, extracting this information is **optimized and efficient**.

---



📌 **Explanation:**  
- Creates a DataFrame with **OHLC price data**  
- Uses `yang_zhang_volatility()` to **compute volatility**  

---

## 🔍 Key Functions in Quantreo

### 📌 **Volatility Estimators**
- `yang_zhang_volatility(df, window_size=30)`: Advanced **Yang-Zhang** volatility  
- `rogers_satchell_volatility(df, window_size=30)`: **Intraday** volatility measurement  
- `parkinson_volatility(df, window_size=30)`: **High-Low range-based** volatility  

### 📌 **Transformations & Derivatives**
- `log_pct(df, col, n)`: **Logarithmic returns**  
- `derivatives(df, col)`: **First & second derivative calculations**  

### 📌 **Market Memory & Correlations**
- `auto_corr(df, col, n, lag)`: **Rolling autocorrelation**  
- `hurst(df, col, window)`: **Hurst exponent calculation**  

---

## 🎯 Coming Soon: Target Engineering!

Quantreo will soon include **Target Engineering** functionalities.

✔ Define **robust target variables** for supervised learning  
✔ Adapt target definitions to **various trading strategies**  
✔ Handle **prediction horizons** and complex **labeling** mechanisms  

📢 *Stay tuned for the next updates!*

---

## 🎓 Learn More

📚 Check out our tutorials:  
- [Beginner’s Guide](https://www.quantreo.com)  
- [Advanced Feature Engineering](https://www.quantreo.com/features)  

🚀 **Ready to start? Install Quantreo and try it now!**  
