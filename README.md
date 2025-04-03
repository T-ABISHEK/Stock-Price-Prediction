# Stock Price Prediction

This project is a machine learning-based stock price prediction system built with TensorFlow and Scikit-Learn. It fetches historical stock data, trains an LSTM model, and predicts future stock prices.

## Features
- Fetches historical stock price data
- Preprocesses data and scales it using MinMaxScaler
- Trains an LSTM model for time series forecasting
- Saves and loads trained models for future predictions
- Visualizes predicted stock prices

## Installation

Clone the repository:
```sh
git clone https://github.com/T-ABISHEK/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

Run the training script:
```sh
python main.py
```

This will train the model and predict stock prices for the next 30 days.
