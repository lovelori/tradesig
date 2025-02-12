# main.py

from src.agents.trading_agent import TradingAgent
from src.environments.trading_env import TradingEnv
from src.data.data_loader import DataLoader
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backtest(trading_agent, test_data, initial_balance=10000):
    """Run backtest on test data with trained model"""
    env = TradingEnv(test_data, initial_balance=initial_balance)
    obs, _ = env.reset()
    done = False
    trades = []
    portfolio_values = []
    
    while not done:
        action, _ = trading_agent.model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        current_price = float(test_data.iloc[env.current_step-1]['close'])
        portfolio_value = env.balance + (env.position * current_price)
        
        if action != 0:  # If trade occurred
            trades.append({ 
                'step': env.current_step-1,
                'price': current_price,
                'action': action ,
                'position': env.position,
                'balance': env.balance,
                'portfolio_value': portfolio_value
            })
        
        portfolio_values.append(portfolio_value)
    
    return trades, portfolio_values

def main():
    try:
        # Initialize data loader with an exchange
        data_loader = DataLoader(data_source='binance')
        
        # Load market data
        market_data = data_loader.load_data()
        
        if market_data is None or market_data.empty:
            logger.error("Failed to load market data")
            return
            
        logger.info(f"Loaded {len(market_data)} data points")

        # Initialize trading environment   
        trading_env = TradingEnv(market_data)
        
        # Initialize trading agent
        trading_agent = TradingAgent(trading_env)

        # Train the agent
        num_episodes = 50
        total_timesteps = len(market_data) * num_episodes
        logger.info(f"Starting training for {num_episodes} episodes ({total_timesteps} timesteps)")
        trading_agent.train(total_timesteps=total_timesteps)

        # Save the trained model
        model_path = "models/trading_agent_model"
        trading_agent.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        # Prepare test data (last 20% of the dataset)
        split_idx = int(len(market_data) * 0.8)
        test_data = market_data.iloc[split_idx:]
        
        # Run backtest
        logger.info("Starting backtest...")
        trades, portfolio_values = backtest(trading_agent, test_data)
        
        # Calculate and display results
        initial_value = 10000
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        logger.info(f"\nBacktest Results:")
        logger.info(f"Number of trades: {len(trades)}")
        logger.info(f"Initial portfolio value: ${initial_value:,.2f}")
        logger.info(f"Final portfolio value: ${final_value:,.2f}")
        logger.info(f"Total return: {total_return:.2f}%")
        
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv("backtest_trades.csv", index=False)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()