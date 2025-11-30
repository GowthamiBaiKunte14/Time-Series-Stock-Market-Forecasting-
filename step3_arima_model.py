import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# ---------- SETTINGS ----------
TICKER = "RELIANCE.NS"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# ---------- DOWNLOAD DATA ----------
df = yf.download(TICKER, start=START_DATE, end=END_DATE)
df = df[["Close"]]

# ---------- TRAIN-TEST SPLIT ----------
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

train_series = train["Close"]
test_series = test["Close"]

# ---------- FIT ARIMA MODEL ----------
print("Training ARIMA model...")
model = ARIMA(train_series, order=(5, 1, 0))
model_fit = model.fit()

print(model_fit.summary())

# ---------- FORECAST ----------
print("\nMaking predictions...")
forecast = model_fit.forecast(steps=len(test_series))

# ---------- EVALUATE ----------
rmse = sqrt(mean_squared_error(test_series, forecast))
print(f"\nARIMA RMSE: {rmse:.2f}")

# ---------- PLOT ----------
plt.figure(figsize=(10,5))
plt.plot(train_series.index, train_series, label="Train")
plt.plot(test_series.index, test_series, label="Test")
plt.plot(test_series.index, forecast, label="ARIMA Forecast")
plt.title(f"{TICKER} ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
