import torch
from torch.utils.data import Dataset
import numpy as np

class CryptoDataset(Dataset):
    def __init__(self, df, sequence_length=99):
        """
        Create sequences of data for training
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            sequence_length (int): Number of time steps in each sequence
        """
        self.sequence_length = sequence_length
        data = df[['open', 'high', 'low', 'close', 'volume']].values
        delta = df['delta'].values
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(data) - sequence_length):
            self.sequences.append(data[i:i + sequence_length])
            self.targets.append(delta[i + sequence_length])
            
        self.sequences = torch.FloatTensor(self.sequences)
        self.targets = torch.FloatTensor(self.targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]