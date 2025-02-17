import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from models.loss_functions import DeltaBasedLoss
from models.torch_net import TorchNet
from data.data_loader import DataLoader as CryptoDataLoader
from data.dataset import CryptoDataset

def main(symbol='ETH/USDT'):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    window_size=32
    # Initialize model and move to GPU
    model = TorchNet(window_size).to(device)
    criterion = DeltaBasedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    # Get market data
    data_loader = CryptoDataLoader(data_source='binance', symbol=symbol)
    market_data = data_loader.load_data(limit=15000)
    print(f"Loaded {len(market_data)} data points for {symbol}")
    
    # Split dataset
    train_size = int(0.9 * len(market_data))
    test_size = len(market_data) - train_size
    train_dataset = CryptoDataset(market_data.iloc[:train_size],window_size)
    test_dataset = CryptoDataset(market_data.iloc[train_size:],window_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_size)
    test_loader = DataLoader(test_dataset, batch_size=test_size)
    
    num_epochs = 100
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            # Move batch to GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Testing phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # Move batch to GPU
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f'models/best_model_{symbol.replace("/", "_")}.pth')

if __name__ == '__main__':
    symbols = [
         #'ORDI/USDT',
         #'LINK/USDT',
         #'ETH/USDT',
         #'DOGE/USDT',
         #'AAVE/USDT',
        #  'NEAR/USDT',
        #  'SOL/USDT',
         # 'BTC/USDT',
         # 'BNB/USDT',
         # 'AVAX/USDT',
        #  'DOT/USDT',
         'ARB/USDT',
        
         #'CRV/USDT',
        # # 'ARB/USDT',
        # # 'OP/USDT',
         'UNI/USDT',
    ]
    for symbol in symbols:
        main(symbol)
