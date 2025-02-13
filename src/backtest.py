import torch
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import DataLoader
from models.torch_net import TorchNet
from data.dataset import CryptoDataset
import os

class Backtester:
    def __init__(self, initial_capital=1000):
        self.capital = initial_capital  # Cash
        self.position = 0  # Crypto holdings
        self.total_value_history = []
        self.price_history = []
        self.signal_history = []  # Add signal history

    def execute_trade(self, signal, current_price):
        """
        Execute trade based on model signal
        signal: float between -1 and 1
        """
        if current_price <= 0:
            return
        if signal > 0:  # Buy signal
            buy_amount = self.capital * abs(signal/2)
            self.position +=  0.9995*buy_amount / current_price
            self.capital -= buy_amount
        elif signal < 0:  # Sell signal
            sell_amount = self.position * abs(signal/2) 
            self.position -= sell_amount
            self.capital += 0.9995*sell_amount * current_price

    def get_total_value(self, current_price):
        return self.capital + (self.position * current_price)

def main(symbol='DOGE/USDT'):
    # Load the trained model
    model = TorchNet()
    model_filename = f'models/{symbol.replace("/", "_")}_model.pth'
    
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"No trained model found for {symbol}. Please train the model first.")
    
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    # Get market data
    data_loader = DataLoader(data_source='binance', symbol=symbol)
    market_data = data_loader.load_data()
    
    # Only use the most recent 50% of data
    half_point = len(market_data) // 3
    market_data = market_data.iloc[half_point:]
    
    # Setup backtester
    backtester = Backtester(initial_capital=1000)
    sequence_length = 99

    # Prepare price data
    prices = market_data['close'].values
    
    for i in range(sequence_length, len(prices)):
        # Prepare input sequence
        sequence = market_data[['open', 'high', 'low', 'close', 'volume']].values[i-sequence_length:i]
        sequence = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Get model prediction
        with torch.no_grad():
            signal = model(sequence).item()
            signal = np.clip(signal, -1, 1)  # Clip signal to [-1, 1]
            backtester.signal_history.append(signal)  # Record signal
        
        # Execute trade
        current_price = prices[i]
        backtester.execute_trade(signal, current_price)
        
        # Record total value
        total_value = backtester.get_total_value(current_price)
        backtester.total_value_history.append(total_value)
        backtester.price_history.append(current_price)
        
        if i % 100 == 0:
            print(f"Processing step {i}/{len(prices)}, Total Value: {total_value:.2f}")

    # Create subplots for better visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Normalize prices for buy & hold comparison
    initial_price = prices[sequence_length]
    normalized_prices = prices[sequence_length:] / initial_price * 1000  # Normalize to initial capital

    # Plot portfolio value and buy & hold comparison
    ax1.plot(backtester.total_value_history, label='Portfolio Value')
    ax1.plot(normalized_prices, label='Buy & Hold', alpha=0.7)
    ax1.set_title(f'Backtesting Results for {symbol.replace("/", "_")}')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot trading signals
    ax2.plot(backtester.signal_history, label='Trading Signals', color='red')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(range(len(backtester.signal_history)), 
                     backtester.signal_history,
                     0, 
                     where=(np.array(backtester.signal_history) > 0),
                     color='green', 
                     alpha=0.3,
                     label='Buy Signal')
    ax2.fill_between(range(len(backtester.signal_history)), 
                     backtester.signal_history,
                     0, 
                     where=(np.array(backtester.signal_history) < 0),
                     color='red', 
                     alpha=0.3,
                     label='Sell Signal')
    ax2.set_title('Trading Signals')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Signal Strength')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'backtest_results_{symbol.replace("/", "_")}.png')
    plt.show()

    # Print final statistics
    initial_price = prices[sequence_length]
    final_price = prices[-1]
    buy_hold_return = (final_price - initial_price) / initial_price * 100
    strategy_return = (backtester.total_value_history[-1] - 1000) / 1000 * 100
    
    print(f"\nBacktesting Results:",symbol)
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Strategy Return: {strategy_return:.2f}%")
    print(f"Final Portfolio Value: ${backtester.total_value_history[-1]:.2f}")

if __name__ == '__main__':
    import sys
    symbols = [
        'BTC/USDT',
        'BNB/USDT',
        'LTC/USDT',
        'LINK/USDT',
        'NEAR/USDT',
        'ETH/USDT',
        "DOGE/USDT"
    ]

    for symbol in symbols:
        main(symbol)