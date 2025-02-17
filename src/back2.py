import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.torch_net import TorchNet
from data.data_loader import DataLoader
import torch
import os

class Portfolio:
    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position = 0
        self.history = []
        
    def get_total_value(self, current_price):
        return self.balance + self.position * current_price
    
    def buy(self, price, signal):
        """Buy based on positive signal strength"""
        amount = self.balance * (abs(signal) * 0.4)  # Use up to 50% of balance
        if amount > 0:
            shares = amount / price
            self.position += shares
            self.balance -= amount
            
    def sell(self, price, signal):
        """Sell based on negative signal strength"""
        if self.position > 0:
            shares = self.position * (abs(signal) * 0.4)  # Sell up to 50% of position
            amount = shares * price
            self.position -= shares
            self.balance += amount

def backtest(symbol='ETH/USDT', plot=True):
    # Load model
    model = TorchNet()
    model_filename = f'models/best_model_{symbol.replace("/", "_")}.pth'
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"No trained model found for {symbol}")
    
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    # Get market data
    data_loader = DataLoader(data_source='binance', symbol=symbol)
    market_data = data_loader.load_data().iloc[-1000:]
    
    # Initialize portfolio
    portfolio = Portfolio()
    sequence_length = 32
    
    # Store history
    portfolio_values = []
    prices = market_data['close'].values
    signals = []
    
    # Run backtest
    for i in range(sequence_length, len(prices)):
        current_price = prices[i]
        
        # Prepare input sequence
        sequence = market_data[['volume', 'high', 'low', 'close']].values[i-sequence_length:i]
        sequence = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            signal = model(sequence).item()
            signals.append(signal)
            
            if signal > 0:
                portfolio.buy(current_price, signal)
            elif signal < 0:
                portfolio.sell(current_price, signal)
        
        portfolio_values.append(portfolio.get_total_value(current_price))
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot portfolio value
        ax1.plot(portfolio_values, label='Portfolio Value', color='blue')
        ax1.set_title(f'{symbol} Portfolio Value')
        ax1.set_ylabel('USD')
        ax1.grid(True)
        ax1.legend()
        
        # Plot price and signals
        ax2.plot(prices[sequence_length:], label='Price', color='gray', alpha=0.6)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(signals, label='Signals', color='red', alpha=0.5)
        ax2.set_title('Price and Signals')
        ax2.set_ylabel('Price')
        ax2_twin.set_ylabel('Signal Strength')
        ax2.grid(True)
        
        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        final_value = portfolio_values[-1]
        returns = (final_value - portfolio.initial_capital) / portfolio.initial_capital * 100
        print(f"\nBacktest Results for {symbol}:")
        print(f"Initial Capital: ${portfolio.initial_capital:.2f}")
        print(f"Final Value: ${final_value:.2f}")
        print(f"Return: {returns:.2f}%")
    
    return portfolio_values, signals

if __name__ == '__main__':
    symbols = [
        # 'ETH/USDT',
        # 'LINK/USDT',
        'DOGE/USDT',
        'AAVE/USDT',
    ]
    
    for symbol in symbols:
        backtest(symbol)