import yfinance as yf
import matplotlib.pyplot as plt

# ---------- SETTINGS ----------
TICKER = "RELIANCE.NS"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# ---------- DOWNLOAD DATA ----------
df = yf.download(TICKER, start=START_DATE, end=END_DATE)
df = df[["Close"]]  # Keep only Close price

# Split into train (80%) and test (20%)
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

print("Train size:", train.shape)
print("Test size:", test.shape)

# ---------- PLOT ----------
plt.figure(figsize=(10,5))
plt.plot(train.index, train["Close"], label="Train Data")
plt.plot(test.index, test["Close"], label="Test Data")
plt.title(f"{TICKER} Train-Test Split")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

