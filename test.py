from quantreo.features_engineering.volatility import *
from datetime import datetime
from quantreo.features_engineering.math import *


df = pd.read_parquet("Data/ML_Strategy_4H_EURUSD.parquet")

start = datetime.now()

df = derivatives(df, "close")
end = datetime.now()
print((end-start).total_seconds(), "s")
print(df)