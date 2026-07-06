import sqlite3
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class FinancialDataset(Dataset):
    def __init__ (self, db_path ,ticker, window_size = 20):
        self.window_size = window_size

        #1. load data from db
        conn = sqlite3.connect(db_path)
        extract_data_query = f"select date, open, high , low , close, volume from daily_stock_prices where ticker = '{ticker}' order by date asc"
        df = pd.read_sql(extract_data_query, conn)
        conn.close()

        # print(df.shape)
        # print(df.head())

        #2. moving average -> NAN for first few data due to window
        df['ma_5'] = df['close'].rolling(window=5).mean()
        # print(df['close'].shape)
        # print(df['ma_5'].head())
        df['ma_20'] = df['close'].rolling(window=20).mean()

        #3.RSI calculation (14days ) -> NAN for first data due to diff
        # -> type -> series
        delta = df['close'].diff()
        # print(type(delta))
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        #14days based average gain and loss
        A_G = up.rolling(window=14).mean()
        A_L = down.rolling(window=14).mean()

        #Relative Strength -> RS
        RS = A_G / (A_L + 1e-10)
        #Relative Strength Index -> RSI
        df['rsi'] = 100 - (100/ (1 + RS))

        #RSI >= 70 -> time to sell(Overbought) , RSI <= 30 -> time to buy (Oversold)

        #4 Design deeplearning input features
        price_cols = ['open', 'high', 'low', 'close']
        moving_average_cols = ['ma_5', 'ma_20']
        df[price_cols] = df[price_cols].pct_change()
        df[moving_average_cols] = df[moving_average_cols].pct_change()
        df['volume'] = df['volume'].pct_change()
        df['rsi'] = df['rsi'] / 100.0
        print(len(df[df['rsi'] >= 0.7]))

        #first few data will be lost ->  
        # 내가 궁금한점: Does losing the first few data points actually affect the performance of our predictive model in practice?
        df = df.dropna().reset_index(drop = True)
        print(df)
        feature_cols = ['open', 'high', 'low', 'close', 'ma_5', 'ma_20', 'volume' ,'rsi']
        print(df[feature_cols].shape)
        #change to pytorch tensor (input features for my deep learning model)
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df['close'].values, dtype=torch.float32)
        print(f'features shape is {self.features.shape}')
        print(self.labels.shape)

    def __len__(self):
        return len(self.features) - self.window_size
    def __getitem__(self, idx):
        #
        X = self.features[idx: idx + self.window_size]

        y = self.labels[idx + self.window_size]
        return X , y
        


#Raw data(DB) -> pandas(5011, 8) -> Dataset(__getitem__) -> DataLoader(Batch) -> Transformer_model input[32,20,8]
if __name__ == "__main__":
    import os
    db_path = "data/financial_market.db" 
    
    if os.path.exists(db_path):
        #make custom dataset we created
        dataset = FinancialDataset(db_path, ticker="AAPL", window_size=20)
        #32 batch size for gpu processing , shuffle=False to prevent breaking time sequence
        dataloader = DataLoader(dataset, batch_size = 32, shuffle=False)
        #we have to think about last batch
        for X_batch, y_batch in dataloader:
            print(f"deeplearning input (X_batch) : {X_batch.shape}") #[32, 20, 8]
            print(f"deeplearning label (y_batch) : {y_batch.shape}") #[32]
            break 