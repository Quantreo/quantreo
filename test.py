from quantreo.features_engineering.volatility import *
from datetime import datetime
from quantreo.features_engineering.market_regime import *
import ta

df = pd.read_parquet("Data/ML_Strategy_4H_EURUSD.parquet")
print(df)
start = datetime.now()

dft = kama_market_regime(df,"close", 30, 100)


end = datetime.now()
print((end-start).total_seconds(), "s")
print(dft)


"""start = datetime.now()
r = hurst_b(df, "close", 100)


end = datetime.now()
print((end-start).total_seconds(), "s")
print(r)"""