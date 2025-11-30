import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

from prophet import Prophet

# ---------- SETTINGS ----------
TICKER = "RELIANCE.NS"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# ---------- DOWNLOAD DATA ----------
print("Downloading data...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

if df.empty:
    raise ValueError("No data downloaded. Check TICKER or dates.")

# ---------- TAKE CLOSE AS A 1-D SERIES ----------
# df[['Close']] -> DataFrame, .squeeze() -> Series
close_series = df[['Close']].squeeze()

# Just to be sure it is a Series
print("\nType of close_series:", type(close_series))
print("close_series shape:", close_series.shape)

# ---------- BUILD CLEAN DATAFRAME FOR PROPHET ----------
df_prophet = pd.DataFrame({
    "ds": close_series.index,                          # dates
    "y": close_series.to_numpy(dtype="float64")        # values as 1-D float array
})

print("\nSample df_prophet:")
print(df_prophet.head())
print(df_prophet.dtypes)

# ---------- TRAIN-TEST SPLIT ----------
split_index = int(len(df_prophet) * 0.8)
train = df_prophet.iloc[:split_index]
test = df_prophet.iloc[split_index:]

print("\nTrain size:", train.shape)
print("Test size :", test.shape)

# ---------- FIT PROPHET ----------
print("\nTraining Prophet model...")
m = Prophet(daily_seasonality=True)
m.fit(train)

# ---------- FORECAST ----------
future = m.make_future_dataframe(periods=len(test), freq="D")
forecast = m.predict(future)

# Take only the forecast rows corresponding to test period
forecast_test = forecast.iloc[-len(test):]

# ---------- EVALUATE ----------
rmse_prophet = sqrt(mean_squared_error(test["y"], forecast_test["yhat"]))
print(f"\nProphet RMSE: {rmse_prophet:.2f}")

# ---------- PLOTS ----------
# 1) Full forecast
m.plot(forecast)
plt.title(f"{TICKER} - Prophet Full Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# 2) Test vs forecast
plt.figure(figsize=(10, 5))
plt.plot(test["ds"], test["y"], label="Actual Test Data")
plt.plot(test["ds"], forecast_test["yhat"], label="Prophet Forecast")
plt.title(f"{TICKER} - Prophet Test Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

