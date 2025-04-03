import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from src.data_loader import DataLoader

class StockPredictor:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.model = tf.keras.models.load_model("models/stock_model.keras")
        self.scaler = joblib.load("models/scaler.pkl")

    def predict_future(self, steps=30):
        """Predicts future stock prices for given steps."""
        loader = DataLoader(self.stock_symbol, start_date="2020-01-01", end_date="2024-01-01")
        df = loader.fetch_data()
        
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  
        df.dropna(inplace=True)  
        data = df[['Close']].values

        data_scaled = self.scaler.transform(data) 

        last_60_days = data_scaled[-60:].reshape(1, -1, 1) 
        predictions = []

        for _ in range(steps):
            pred_scaled = self.model.predict(last_60_days)
            pred_rescaled = self.scaler.inverse_transform(pred_scaled) 
            predictions.append(pred_rescaled[0][0])

            last_60_days = np.append(last_60_days[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)

        return predictions
