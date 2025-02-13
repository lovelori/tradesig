import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.torch_net import TorchNet
from data.data_loader import DataLoader as CryptoDataLoader

def load_model(model_path):
    model = TorchNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def backtest(df, model, window_size=100, initial_capital=1000):
    """
    Backtest the trading strategy with separate tracking of position and balance
    """
    portfolio_values = [initial_capital]
    balance = initial_capital  # Cash balance
    position = 0  # Number of coins held
    
    # Get features for testing
    features = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
    
    for i in range(window_size, len(df)):
        try:
            # Get the window of data
            window = features[i-window_size:i]
            x = window.T
            x = torch.tensor(x).unsqueeze(0)
            
            # Get model prediction
            with torch.no_grad():
                action = model(x).item()  # Value between -1 and 1
            
            current_price = features[i, 3]  # Current close price
            portfolio_value = balance + position * current_price
            #print(current_price)
            # Execute trades based on action
            print(action)
            if action > 0:  # Buy
                buy_amount = balance * action
                new_position = buy_amount / current_price
                balance -= buy_amount
                position += new_position
            elif action < 0:  # Sell
                sell_ratio = abs(action)
                sell_position = position * sell_ratio
                sell_amount = sell_position * current_price
                balance += sell_amount
                position -= sell_position
                
            # Calculate new portfolio value
            portfolio_value = balance + position * current_price
            portfolio_values.append(portfolio_value)
            
        except Exception as e:
            print(f"Error at index {i}: {str(e)}")
            portfolio_values.append(portfolio_values[-1])
            continue
            
    return portfolio_values, balance, position

def plot_results(df, portfolio_values):
    """
    Plot the portfolio value and price curves
    """
    plt.figure(figsize=(12, 6))
    
    # Ensure x-axis dates and portfolio values have same length
    dates = df.index[98:len(portfolio_values)+99]  # Adjust date range to match portfolio values
    
    # Plot portfolio value
    plt.plot(dates, portfolio_values, label='Portfolio Value', color='blue')
    
    # Plot price for comparison (normalized to start at initial_capital)
    price_series = df['close'].values[99:len(portfolio_values)+99]  # Match length with portfolio values
    initial_price = price_series[0]
    normalized_prices = price_series * (1000 / initial_price)
    #plt.plot(dates, normalized_prices, label='Buy & Hold', color='gray', alpha=0.6)
    
    plt.title('Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load the saved model
    model = load_model('best_model.pth')
    
    # Load the same data used for training
    data_loader = CryptoDataLoader(
        data_source='binance',
        symbol='ETH/USDT',
        timeframe='4h'
    )
    
    # Load all data
    df = data_loader.load_data(
        limit=15000,
        use_cache=True,
        normalize=True
    )
    
    # Split data - use last 30% for backtesting
    split_idx = int(len(df) * 0.5)  # 70% train, 30% test
    test_df = df.iloc[split_idx:]
    
    # Run backtest on test data
    portfolio_values, final_balance, final_position = backtest(test_df, model, window_size=99, initial_capital=1000)
    
    # Calculate metrics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100
    print(f"Backtest Period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Final Position: {final_position:.6f} coins")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
    
    # Plot results
    plot_results(test_df, portfolio_values)