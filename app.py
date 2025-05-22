import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime
from streamlit_lottie import st_lottie
import requests

# Set background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1621375271940-2b50818c6eb4");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

add_bg_from_url()
crypto_anim = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")

st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🔮 Crypto Price Predictor using LSTM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict future prices of your favorite cryptocurrencies using AI-powered LSTM models.</p>", unsafe_allow_html=True)

st_lottie(crypto_anim, height=200)

crypto_names = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Dogecoin": "DOGE-USD",
    "Cardano": "ADA-USD",
    "Solana": "SOL-USD",
    "Binance Coin": "BNB-USD",
    "Ripple (XRP)": "XRP-USD",
    "Polkadot": "DOT-USD",
    "Litecoin": "LTC-USD",
    "Chainlink": "LINK-USD"
}

crypto_choice = st.selectbox("Choose a Cryptocurrency", list(crypto_names.keys()))
ticker = crypto_names[crypto_choice]

start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

if st.button("Train & Predict"):

    st.info("Fetching data...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found. Try a valid crypto or different date range.")
        st.stop()

    st.write("Sample Data", df.head())

    data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 3])

    X, y = np.array(X), np.array(y)

    st.info("Training LSTM model...")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    st.info("Predicting future crypto prices...")
    test_data = yf.download(ticker, start="2024-01-01", end="2025-01-01")
    if test_data.empty:
        st.error("No test data available for prediction.")
        st.stop()

    total = pd.concat((df, test_data), axis=0)
    inputs = total[['Open', 'High', 'Low', 'Close', 'Volume']].values
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(len(df), len(inputs)):
        X_test.append(inputs[i-sequence_length:i])
    X_test = np.array(X_test)

    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(
        np.hstack((np.zeros((predicted_prices.shape[0], 3)), predicted_prices, np.zeros((predicted_prices.shape[0], 1))))
    )[:, 3]

    actual_prices = test_data['Close'].values

    st.subheader("📈 Predict Future Crypto Prices")
    n_days = st.slider("Select number of future days to predict", 1, 30, 7)

    last_sequence = scaled_data[-sequence_length:]
    future_predictions = []

    for _ in range(n_days):
        input_seq = last_sequence.reshape(1, sequence_length, X.shape[2])
        pred = model.predict(input_seq, verbose=0)[0][0]
        dummy_row = [0, 0, 0, pred, 0]
        future_predictions.append(pred)
        last_sequence = np.vstack((last_sequence[1:], dummy_row))

    future_prices = scaler.inverse_transform(
        np.hstack((np.zeros((n_days, 3)), np.array(future_predictions).reshape(-1,1), np.zeros((n_days,1))))
    )[:, 3]

    future_dates = pd.date_range(start=test_data.index[-1] + pd.Timedelta(days=1), periods=n_days)

    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_prices})
    future_df.set_index('Date', inplace=True)
    st.line_chart(future_df)
    st.write(future_df)

    st.subheader("📊 Predicted vs Actual Crypto Prices")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual_prices, label="Actual Price", color='black')
    ax.plot(predicted_prices, label="Predicted Price", color='green')
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)