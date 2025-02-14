import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from models.loss_functions import train_model
from models.torch_net import TorchNet
from data.data_loader import DataLoader as CryptoDataLoader
from data.dataset import CryptoDataset

def main(symbol='ETH/USDT'):
    # Initialize model
    model = TorchNet()
    
    # Get market data
    data_loader = CryptoDataLoader(data_source='binance', symbol=symbol)
    market_data = data_loader.load_data()
    print(f"Loaded {len(market_data)} data points for {symbol}")
    
    # Create PyTorch dataset
    dataset = CryptoDataset(market_data)
    
    # Create PyTorch DataLoader
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=dataset.__len__(),
        shuffle=True,
        num_workers=0  # For Windows, start with 0
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    # Training parameters
    num_epochs = 320
    
    # Train the model
    train_model(model, train_loader, optimizer, num_epochs)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the trained model with symbol name
    model_filename = f'models/{symbol.replace("/", "_")}_model.pth'
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == '__main__':
    symbols = [
        'LTC/USDT',
        'LINK/USDT',
        'ETH/USDT',
        'DOGE/USDT',
        # 'NEAR/USDT',
        # 'SOL/USDT',
         'AAVE/USDT',
        # 'AVAX/USDT',
        # 'DOT/USDT',
        # 'CRV/USDT',
        # 'ARB/USDT',
    ]
    for symbol in symbols:
        main(symbol)
        