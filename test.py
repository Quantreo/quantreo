from quantreo.features_engineering.volatility import *
from datetime import datetime


df = pd.read_parquet("Data/ML_Strategy_4H_EURUSD.parquet")

start = datetime.now()

df = moving_yang_zhang_estimator(df,30)
end = datetime.now()
print((end-start).total_seconds())
print(df)