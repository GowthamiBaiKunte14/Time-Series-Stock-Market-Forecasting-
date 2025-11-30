import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# -------- SETTINGS (you can change these later) --------
TICKER = "RELIANCE.NS"       # Stock symbol
START_DATE = "2018-01-01"    # From this date
END_DATE = "2024-12-31"      # To this date

print("Downloading data...")

# -------- 1. DOWNLOAD DATA --------
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Check if data is empty
if df.empty:
    print("No data downloaded. Please check the TICKER or date range.")
    exit()

print("Data downloaded successfully!\n")

# -------- 2. SHOW FIRST ROWS --------
print("First 5 rows:")
print(df.head())

# -------- 3. KEEP ONLY 'Close' PRICE --------
close_prices = df["Close"]

print("\nSummary statistics of Close prices:")
print(close_prices.describe())

# -------- 4. PLOT CLOSE PRICE --------
plt.figure(figsize=(10, 5))
plt.plot(close_prices.index, close_prices.values, label="Close Price")
plt.title(f"{TICKER} Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
