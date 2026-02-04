# 📈 Stock Market Price Prediction using LSTM

This project is an **internship-level Stock Market Price Prediction system** built using **LSTM (Long Short-Term Memory)** neural networks. It predicts stock closing prices using **live data fetched from Yahoo Finance** and provides an interactive visualization through a **Streamlit web application**.

---

## 📌 Project Overview
The objective of this project is to analyze historical stock price data and predict future stock prices using deep learning techniques. The system dynamically fetches live stock data from **Yahoo Finance** and applies an LSTM model to learn time-series patterns.

The project also includes an interactive **Streamlit-based web interface** for visualization and user input.

---

## 🧠 Technologies Used
- Python  
- TensorFlow / Keras  
- LSTM (Deep Learning)  
- Yahoo Finance (`yfinance`)  
- Streamlit  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## 📊 Features
- Fetches **live stock data** from Yahoo Finance  
- Supports multiple stock symbols (e.g., GOOG, AAPL, TSLA)  
- Visualizes:
  - Closing price trends  
  - 50, 100, and 200-day moving averages  
  - Actual vs Predicted stock prices  
- Displays performance metrics:
  - RMSE (Root Mean Square Error)  
  - MAE (Mean Absolute Error)  
- Interactive web interface using Streamlit  

---

## 📁 Project Structure
```
Stock-Market-price-prediction/
│
├── app.py
├── Stock_Market_Prediction_Model_Creation.ipynb
├── Stock Predictions Model.keras
└── README.md
```

---

## ⚙️ Requirements
Install the required dependencies using:
```bash
pip install numpy pandas yfinance matplotlib scikit-learn tensorflow streamlit
```

---

## ▶️ How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/balraj-90/Stock-Market-price-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Stock-Market-price-prediction
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

4. Enter a stock symbol and select the date range to view predictions.

---

## 📈 Model Details
- Model Type: LSTM (Recurrent Neural Network)  
- Input: Previous 100 days of closing prices  
- Data Scaling: MinMaxScaler  
- Train-Test Split: 80% training, 20% testing  
- Output: Predicted stock closing prices  

---

## 📉 Performance Metrics
- **RMSE** – Measures the magnitude of prediction errors  
- **MAE** – Measures average absolute prediction error  

---

## 🎓 Internship Learning Outcomes
- Time series forecasting using LSTM  
- Data preprocessing and scaling  
- Sequence generation for RNN models  
- Stock market data analysis  
- Model evaluation using error metrics  
- Deployment of ML models using Streamlit  

---

## ⚠️ Disclaimer
This project is developed **for educational and learning purposes only**.  
It should not be used for real-world trading or financial decision-making.

---

## 👤 Author
**Balraj Jagtap**

---

⭐ If you find this project useful, feel free to star the repository!
