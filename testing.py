import numpy as np
import pandas as pd



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

import quantreo.features_engineering as fe

# Compute Yang-Zhang Volatility with a 3-period rolling window
df["yang_zhang_vol"] = fe.volatility.yang_zhang_volatility(df, window_size=3)

print(df)