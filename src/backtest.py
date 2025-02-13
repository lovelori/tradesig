import torch
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import DataLoader
from models.torch_net import TorchNet
from data.dataset import CryptoDataset

class Backtester:
    def __init__(self, initial_capital=1000):
        self.capital = initial_capital  # Cash
        self.position = 0  # Crypto holdings
        self.total_value_history = []
        self.price_history = []

    def execute_trade(self, signal, current_price):
        """
        Execute trade based on model signal
        signal: float between -1 and 1
        """
        if signal > 0:  # Buy signal
            buy_amount = self.capital * abs(signal)
            self.position += buy_amount / current_price
            self.capital -= buy_amount
        elif signal < 0:  # Sell signal
            sell_amount = self.position * abs(signal) 
            self.position -= sell_amount
            self.capital += sell_amount * current_price

    def get_total_value(self, current_price):
        return self.capital + (self.position * current_price)

def main():
    # Load the trained model
    model = TorchNet()
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    # Get market data
    data_loader = DataLoader(data_source='binance')
    market_data = data_loader.load_data()
    
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
        
        # Execute trade
        current_price = prices[i]
        backtester.execute_trade(signal, current_price)
        
        # Record total value
        total_value = backtester.get_total_value(current_price)
        backtester.total_value_history.append(total_value)
        backtester.price_history.append(current_price)
        if i>1400 and i<1500:
            print(f"step {i}, signal: {signal}, current price: {current_price}, total value: {total_value}")
        if i % 100 == 0:
            print(f"Processing step {i}/{len(prices)}, Total Value: {total_value:.2f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(backtester.total_value_history, label='Portfolio Value')
    
    # Normalize price to initial capital for comparison
    normalized_prices = prices[sequence_length:] * (1000 / prices[sequence_length])
    plt.plot(normalized_prices, label='Buy & Hold', alpha=0.7)
    
    plt.title('Backtesting Results')
    plt.xlabel('Time Steps')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtest_results.png')
    plt.show()

    # Print final statistics
    initial_price = prices[sequence_length]
    final_price = prices[-1]
    buy_hold_return = (final_price - initial_price) / initial_price * 100
    strategy_return = (backtester.total_value_history[-1] - 1000) / 1000 * 100
    
    print(f"\nBacktesting Results:")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Strategy Return: {strategy_return:.2f}%")
    print(f"Final Portfolio Value: ${backtester.total_value_history[-1]:.2f}")

if __name__ == '__main__':
    main()