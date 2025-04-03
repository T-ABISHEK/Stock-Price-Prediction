import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import DataLoader
from src.model import build_model
import pandas as pd

class StockTrainer:
    def __init__(self, stock_symbol, start_date, end_date):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self):
        """Loads and preprocesses data."""
        loader = DataLoader(self.stock_symbol, self.start_date, self.end_date)
        df = loader.fetch_data()

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  
        df.dropna(inplace=True)  
        data = df[['Close']].values.astype(float)  

        data_scaled = self.scaler.fit_transform(data)

        X_train, y_train = [], []
        for i in range(60, len(data_scaled)): 
            X_train.append(data_scaled[i-60:i])
            y_train.append(data_scaled[i])
        
        return np.array(X_train), np.array(y_train), df

    def train(self):
        """Trains and saves the model."""
        X_train, y_train, df = self.prepare_data()
        model = build_model((X_train.shape[1], 1))

        model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)
        model.save("models/stock_model.keras")
        joblib.dump(self.scaler, "models/scaler.pkl")  

        print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    trainer = StockTrainer(stock_symbol="AAPL", start_date="2020-01-01", end_date="2024-01-01")
    trainer.train()
