import numpy as np
import matplotlib.pyplot as plt
from src.train import StockTrainer
from src.predict import StockPredictor

def main():
    stock_symbol = "AAPL"  
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    trainer = StockTrainer(stock_symbol, start_date, end_date) 
    trainer.train()

    predictor = StockPredictor(stock_symbol)
    predictions = predictor.predict_future(steps=30)

    future_days = np.arange(1, 31)  # Days 1 to 30
    predicted_prices = [float(p) for p in predictions]

    plt.figure(figsize=(10, 5))
    plt.plot(future_days, predicted_prices, marker='o', linestyle='-', color='blue', label="Predicted Price")
    plt.xlabel("Days in the Future")
    plt.ylabel("Stock Price (USD)")
    plt.title(f"Stock Price Prediction for {stock_symbol}")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
