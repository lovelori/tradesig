import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.loss_functions import train_model
from models.torch_net import TorchNet
from data.data_loader import DataLoader as CryptoDataLoader
from data.dataset import CryptoDataset

def main():
    # Initialize model
    model = TorchNet()
    
    # Get market data
    data_loader = CryptoDataLoader(data_source='binance')
    market_data = data_loader.load_data()
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 300
    
    # Train the model
    train_model(model, train_loader, optimizer, num_epochs)
    
    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == '__main__':
    main()