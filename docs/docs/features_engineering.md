# **Features Engineering**

Feature engineering is a <u>crucial step</u> in **quantitative trading**. The **Quantreo library** provides several optimized functions to extract meaningful insights from raw market data.

To help you understand how to use each function, we provide **detailed explanations** and **Jupyter Notebook examples** in the `examples/` folder.

!!! warning "Warning"
    Be sure you have installed the quantreo library before running any of these codes. To install it, just run ``pip install quantreo`` in your terminal.

!!! danger "Danger"
    <u>**PARLER DU FORMAT DES DONNÉES**</u>

---

## **Candle Information**
You can find a series of examples on how to create these features in the [educational notebooks](./examples/educational_notebooks.ipynb) provided by [Quantreo](https://www.quantreo.com).


``` py title="Module Import"
from quantreo.features_engineering import candle
```

<br>

### **Basic Candle Information**

The `candle_information` function provides essential insights about a candle by calculating the following features:

- **The Candle Way**: This feature returns `-1` if the candle is red (indicating a negative variation from the open price to the close price) or `1` if the candle is green (indicating a positive variation from the open price to the close price).

- **The Filling**: This feature computes the ratio between the absolute difference from the open to the close price and the total range of the candle (i.e., the difference between the high and low prices). $\frac{| \text{close} - \text{open} |}{| \text{high} - \text{low} |}$

- **The Amplitude**: This feature measures the relative price movement of a candle compared to its average. $\frac{| \text{close} - \text{open} |}{\left( \frac{\text{open} + \text{close}}{2} \right)}$


```python
def candle_information(df: pd.DataFrame, open_col: str = 'open', high_col: str = 'high',
                       low_col: str = 'low', close_col: str = 'close') 
                       -> Tuple[pd.Series, pd.Series, pd.Series]:
```

📢 "For a practical example, check out the [educational notebook](to define)."

!!! tip "Tip"
    These three variables (`candle_way`, `filling` and `amplitude`) are often very correlated. Do not hesitate to keep only the best feature related to your problem or combine them smartly into one variable.





<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>






!!! tip "Tip"
    If you don't have prior experience with Python, we recommend reading [Using Python's pip to Manage Your Projects' Dependencies](https://pip.pypa.io/en/stable/user_guide/), which is a really good introduction on the mechanics of Python package management and helps you troubleshoot if you run into errors.


!!! danger "Danger"
    Running this command may delete important files!

!!! note "Note"
    This is a simple note box.

!!! important "Important"
    Make sure you install all dependencies before proceeding.

