import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------- SETTINGS ----------
TICKER = "RELIANCE.NS"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
WINDOW_SIZE = 60   # past 60 days to predict next day

# ---------- 1. DOWNLOAD DATA ----------
print("Downloading data...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

if df.empty:
    raise ValueError("No data downloaded. Check TICKER or dates.")

# Use only Close prices
close_prices = df[["Close"]].values   # shape: (n, 1)

# ---------- 2. SCALE DATA ----------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(close_prices)

# ---------- 3. CREATE SEQUENCES ----------
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])  # past window_size values
        y.append(data[i, 0])                # next value
    X = np.array(X)
    y = np.array(y)
    return X, y

X_all, y_all = create_sequences(scaled, WINDOW_SIZE)
print("Total samples:", X_all.shape[0])

# ---------- 4. TRAIN-TEST SPLIT (80/20) ----------
split_index = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_index], X_all[split_index:]
y_train, y_test = y_all[:split_index], y_all[split_index:]

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape :", X_test.shape, y_test.shape)

# Reshape for LSTM: (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ---------- 5. BUILD LSTM MODEL ----------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# ---------- 6. TRAIN MODEL ----------
print("\nTraining LSTM model (this may take a little time)...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ---------- 7. PREDICT ----------
y_pred_scaled = model.predict(X_test)

# Inverse scale back to original prices
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_reshaped = y_test.reshape(-1, 1)
y_test_actual = scaler.inverse_transform(y_test_reshaped)

# ---------- 8. EVALUATE ----------
rmse_lstm = sqrt(mean_squared_error(y_test_actual, y_pred))
print(f"\nLSTM RMSE: {rmse_lstm:.2f}")

# ---------- 9. PLOT ----------
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label="Actual Test Prices")
plt.plot(y_pred, label="LSTM Predicted Prices")
plt.title(f"{TICKER} - LSTM Forecast (Test Set)")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
