import torch
from torch.utils.data import Dataset
import numpy as np

class CryptoDataset(Dataset):
    def __init__(self, df, sequence_length=20):
        self.df = df
        self.sequence_length = sequence_length
        self.features = ['volume', 'high', 'low', 'close']
        
    def __len__(self):
        return len(self.df) - self.sequence_length - 3  # -5 for future window
        
    def __getitem__(self, idx):
        # Get sequence data for features
        sequence = self.df.iloc[idx:idx + self.sequence_length][self.features].values
        
        # Prepare x: [sequence_length, 4] tensor with volume, high, low, close
        x = torch.FloatTensor(sequence)
        
        # Calculate y: (mean(close[t+1:t+6]) - close[t]) / close[t]
        current_close = self.df.iloc[idx + self.sequence_length - 1]['close']
        
        future_closes = self.df.iloc[idx + self.sequence_length:idx + self.sequence_length + 3]['close'].values
        future_mean = np.mean(future_closes)
        y = (future_mean - current_close) 
        
        return x, torch.FloatTensor([y])