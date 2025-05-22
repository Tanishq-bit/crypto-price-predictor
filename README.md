# 📈 Stock Price Predictor Web App

A real-time, multi-stock **price prediction web app** using **LSTM neural networks**, built with **Streamlit**. This app fetches historical stock data using `yfinance`, trains an LSTM model on multiple features (Open, High, Low, Close, Volume), and predicts both **actual and future stock prices** with interactive visualizations.

---

## 🚀 Features

- 🔍 Select any stock by ticker symbol (e.g., AAPL, TSLA, MSFT)
- 📊 Trains on multiple features (Open, High, Low, Close, Volume)
- 📈 Predicts future stock prices (select days ahead)
- 🧠 LSTM-based time series forecasting
- 📉 Visual comparison of predicted vs actual stock prices
- 🌐 Deployed via Streamlit Cloud (or Render)
---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor

pip install -r requirements.txt
streamlit run app.py

stock-price-predictor/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshot.png  # Optional: insert into README
