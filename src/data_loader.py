import pandas as pd
import yfinance as yf
import os

class DataLoader:
    def __init__(self, stock_symbol, start_date, end_date):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        """Downloads stock data and saves it as a CSV."""
        file_path = f"data/{self.stock_symbol}.csv"

        if os.path.exists(file_path):  
            print(f"Loading existing data for {self.stock_symbol}...")
            return pd.read_csv(file_path, index_col="Date", parse_dates=True)

        print(f"Fetching data for {self.stock_symbol} from Yahoo Finance...")
        stock_data = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)
        stock_data.reset_index(inplace=True)  
        stock_data.to_csv(file_path, index=False)
        return stock_data
