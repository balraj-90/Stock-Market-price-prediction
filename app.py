import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load model
model = load_model('Stock Predictions Model.keras')

# App title
st.title('📈 Stock Market Price Predictor with LSTM')

# User input
stock = st.text_input('Enter Stock Symbol (e.g., GOOG)', 'GOOG')

# Select date range
start = st.date_input("Start Date", pd.to_datetime("2012-01-01"))
end = st.date_input("End Date", pd.to_datetime("2022-12-31"))

# Fetch data
data = yf.download(stock, start, end)

st.subheader('Raw Stock Data')
st.write(data.tail())

# Plot MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig1)

# Plot MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig2)

# Plot MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig3)

# Prepare data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
final_test_data = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_test_data)

# Create sequences
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
predicted_prices = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
predicted_prices = predicted_prices * scale_factor
y_test = y_test * scale_factor

# Plot prediction
st.subheader('📊 Predicted Price vs Original Price')
fig4 = plt.figure(figsize=(10, 6))
plt.plot(predicted_prices, 'r', label='Predicted Price')
plt.plot(y_test, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, predicted_prices))
mae = mean_absolute_error(y_test, predicted_prices)
st.subheader("📈 Model Performance")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAE:** {mae:.2f}")

# Learning outcome section
st.markdown("""
---
### ✅ What I Learned from this Internship
- Time Series Forecasting using LSTM
- Data scaling and sequence preparation
- Visualization with Matplotlib & Streamlit
- Stock market trends via moving averages
- Deploying ML models in an interactive web app
""")

