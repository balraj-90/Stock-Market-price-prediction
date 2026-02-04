"""
Streamlit app for live stock price prediction using an LSTM model.
Internship project – Educational purpose only.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# -------------------- App Config --------------------
st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Market Price Predictor with LSTM")


# -------------------- Load Model Safely --------------------
MODEL_PATH = "Stock Predictions Model.keras"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("❌ Model file not found or failed to load.")
    st.stop()


# -------------------- User Input --------------------
stock = st.text_input("Enter Stock Symbol (e.g., GOOG, AAPL, TSLA)", "GOOG")

start = st.date_input("Start Date", pd.to_datetime("2012-01-01"))
end = st.date_input("End Date", pd.to_datetime("2022-12-31"))

if start >= end:
    st.error("❌ Start date must be earlier than end date.")
    st.stop()


# -------------------- Fetch Stock Data --------------------
@st.cache_data
def load_stock_data(symbol, start_date, end_date):
    return yf.download(symbol, start_date, end_date)


data = load_stock_data(stock, start, end)

if data.empty:
    st.error("❌ No data found for the given stock symbol or date range.")
    st.stop()


st.subheader("📄 Raw Stock Data (Last 5 Rows)")
st.write(data.tail())


# -------------------- Moving Averages --------------------
st.subheader("📊 Price vs Moving Averages")

ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

fig1 = plt.figure(figsize=(10, 6))
plt.plot(data.Close, label="Closing Price")
plt.plot(ma_50, label="MA50")
plt.plot(ma_100, label="MA100")
plt.plot(ma_200, label="MA200")
plt.legend()
st.pyplot(fig1)


# -------------------- Data Preparation --------------------
data_close = data[["Close"]]

train_size = int(len(data_close) * 0.80)
data_train = data_close[:train_size]
data_test = data_close[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

past_100_days = data_train_scaled[-100:]
test_data_scaled = scaler.transform(data_test)

final_data = np.concatenate((past_100_days, test_data_scaled))


# -------------------- Create Sequences --------------------
x_test = []
y_test = []

for i in range(100, final_data.shape[0]):
    x_test.append(final_data[i - 100:i])
    y_test.append(final_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)


# -------------------- Prediction --------------------
predicted_prices = model.predict(x_test)

# Inverse scaling
scale_factor = 1 / scaler.scale_[0]
predicted_prices = predicted_prices * scale_factor
y_test = y_test * scale_factor


# -------------------- Plot Predictions --------------------
st.subheader("📈 Predicted Price vs Actual Price")

fig2 = plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual Price")
plt.plot(predicted_prices, label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
st.pyplot(fig2)


# -------------------- Model Performance --------------------
rmse = np.sqrt(mean_squared_error(y_test, predicted_prices))
mae = mean_absolute_error(y_test, predicted_prices)

st.subheader("📉 Model Performance")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAE:** {mae:.2f}")


# -------------------- Learning Outcomes --------------------
st.markdown("""
---
### ✅ Internship Learning Outcomes
- Time series forecasting using LSTM
- Live data fetching with Yahoo Finance
- Data scaling and sequence generation
- Stock trend analysis using moving averages
- Model evaluation using RMSE and MAE
- Deploying ML models using Streamlit
""")


# -------------------- Disclaimer --------------------
st.warning(
    "⚠️ This project is for educational purposes only and should not be used for real-world trading decisions."
)
