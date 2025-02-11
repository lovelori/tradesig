# main.py

from src.agents.trading_agent import TradingAgent
from src.environments.trading_env import TradingEnv
from src.data.data_loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        total_timesteps = len(market_data) * 10  # 10 episodes
        logger.info(f"Starting training for {total_timesteps} timesteps")
        trading_agent.train(total_timesteps=total_timesteps)

        # Save the trained model
        model_path = "models/trading_agent_model"
        trading_agent.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()