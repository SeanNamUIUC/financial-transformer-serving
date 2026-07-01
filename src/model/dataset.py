import sqlite3
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class FinancialDataset(Dataset):
    def __init__ (self, db_path ,ticker, window_size = 20):
        self.window_size = window_size

        #load data from db
        conn = sqlite3.connect(db_path)
        extract_data_query = f"select date, open, high , low , close, volume from daily_stock_prices where ticker = '{ticker}' order by date asc"
        df = pd.read_sql(extract_data_query, conn)
        conn.close()

        # print(df.shape)
        # print(df.head())

        #moving average -> NAN for first few data due to window
        df['ma_5'] = df['close'].rolling(window=5).mean()
        # print(df['close'].shape)
        # print(df['ma_5'].head())
        df['ma_20'] = df['close'].rolling(window=20).mean()

        #RSI (14days ) -> NAN for first data due to diff
        delta = df['close'].diff()
        # print(delta)



if __name__ == "__main__":
    import os
    db_path = "data/financial_market.db" 
    
    if os.path.exists(db_path):
        dataset = FinancialDataset(db_path, ticker="AAPL", window_size=20)
