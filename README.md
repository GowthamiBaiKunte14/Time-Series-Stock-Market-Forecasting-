# 📈 **Time Series Stock Market Forecasting**

Using **ARIMA, Prophet & LSTM**

This project focuses on forecasting stock market trends using different time series models. Historical stock data was collected from **Yahoo Finance**, preprocessed, visualized, and used to train statistical and deep-learning forecasting models. The performance of **ARIMA**, **Facebook Prophet**, and **LSTM** was compared using RMSE.

---

## 🚀 **Project Overview**

Stock prices change frequently and are influenced by many factors. This project demonstrates how to forecast stock prices using:

* Classical Statistical Models
* Machine Learning
* Deep Learning (LSTM)

The project gives hands-on experience with **data analytics, time series modeling, and model evaluation**.

---

## 🧠 **Models Implemented**

| Model       | Description                                  | RMSE              |
| ----------- | -------------------------------------------- | ----------------- |
| **ARIMA**   | Classical time-series model capturing trends | ~166              |
| **Prophet** | Additive forecasting model by Facebook       | ~272              |
| **LSTM**    | Deep learning model for sequences            | **~42.48 (Best)** |

✔ **LSTM clearly outperformed all models**, proving deep learning works better for non-linear stock patterns.

---

## 📊 **Graphs & Visualizations**

The project includes:

* Stock closing price trends
* Train vs Test split
* ARIMA forecast plot
* Prophet forecast plot
* LSTM actual vs predicted plot

These visualizations help in understanding model performance.

---

## 📦 **Tech Stack**

### **Languages**

* Python

### **Libraries**

* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Statsmodels (ARIMA)
* Prophet
* TensorFlow/Keras (LSTM)
* yFinance

### **Optional**

* Streamlit / Flask (for deployment)

---

## 📚 **Project Structure**

```
Time-Series-Stock-Market-Forecasting/
│── step1_load_data.py
│── step2_train_test_split.py
│── step3_arima_model.py
│── step4_prophet_model.py
│── step5_lstm_model.py
│── README.md
```

---

## 📥 **Dataset Source**

Data collected using `yfinance`:

* Ticker Used: **RELIANCE.NS**
* Date Range: 2018 – 2024
* Columns: Open, Close, High, Low, Volume

---

## ⚙️ **How to Run**

1. Clone the repo

```bash
git clone https://github.com/GowthamiBaiKunte14/Time-Series-Stock-Market-Forecasting.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run each step

```bash
python step1_load_data.py
python step2_train_test_split.py
python step3_arima_model.py
python step4_prophet_model.py
python step5_lstm_model.py
```

---

## 🎯 **Conclusion**

LSTM gave the most accurate predictions due to its ability to learn long-term patterns in stock prices.
This shows that **deep learning is more effective** for stock market forecasting than traditional methods.

---

## 🔮 **Future Enhancements**

* Add technical indicators (RSI, MACD, SMA)
* Create a Streamlit dashboard
* Try advanced deep learning models (GRU, Transformers)
* Predict multiple stocks together
* Deploy the model online
