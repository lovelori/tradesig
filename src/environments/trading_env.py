import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        logger.info(f"Initialized environment with {self.max_steps + 1} data points")
        
        # Define action space (0: hold, 1-4: buy 10-40%, 5-8: sell 10-40%)
        self.action_space = spaces.Discrete(9)
        
        # Define buy and sell percentages (both 10-40%)
        self.buy_percentages = [0.1, 0.2, 0.3, 0.4]  # 10-40%
        self.sell_percentages = [0.1, 0.2, 0.3, 0.4]  # 10-40%
        
        # Define observation space (price data + account info)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,),  # OHLCV + position
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.done = False
        
        logger.debug("Environment reset")
        return self._get_observation(), {}
    
    def step(self, action):
        if self.done:
            logger.warning("Step called on done environment")
            return self._get_observation(), 0, True, False, {}
            
        current_price = float(self.data.iloc[self.current_step]['close'])
        
        # Execute trading action
        reward = 0
        if 1 <= action <= 4 and self.balance > 0:  # Buy actions
            buy_percentage = self.buy_percentages[action - 1]
            buy_amount = self.balance * buy_percentage
            new_position = buy_amount / current_price
            self.position += new_position
            self.balance -= buy_amount
            logger.debug(f"Buy {buy_percentage*100}% at price: {current_price}")
            
        elif 5 <= action <= 8 and self.position > 0:  # Sell actions
            sell_percentage = self.sell_percentages[action - 5]
            sell_position = self.position * sell_percentage
            sell_amount = sell_position * current_price
            self.position -= sell_position
            self.balance += sell_amount
            logger.debug(f"Sell {sell_percentage*100}% at price: {current_price}")
                
        # Move to next step
        self.current_step += 1
        self.done = self.current_step >= self.max_steps
        
        reward = self._calculate_reward()
        observation = self._get_observation()
        
        return observation, reward, self.done, False, {}
    
    def _get_observation(self):
        if self.current_step > self.max_steps:
            logger.warning("Attempting to get observation beyond data range")
            return np.zeros(6, dtype=np.float32)
            
        try:
            current_data = self.data.iloc[self.current_step]
            obs = np.array([
                float(current_data['open']),
                float(current_data['high']),
                float(current_data['low']),
                float(current_data['close']),
                float(current_data['volume']),
                float(self.position)
            ], dtype=np.float32)
            return obs
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            return np.zeros(6, dtype=np.float32)
    
    def _calculate_reward(self):
        if self.current_step == 0:
            return 0.0
            
        try:
            # Calculate portfolio values
            current_price = float(self.data.iloc[self.current_step]['close'])
            prev_price = float(self.data.iloc[self.current_step-1]['close'])
            current_portfolio_value = self.balance + (self.position * current_price)
            prev_portfolio_value = self.balance + (self.position * prev_price)
            
            # Calculate returns
            portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            market_return = (current_price - prev_price) / prev_price
            
            # Calculate Sharpe-like ratio component (excess returns over market)
            excess_return = portfolio_return - market_return
            
            # Add position holding cost (penalize holding positions)
            holding_cost = -0.0001 * abs(self.position)  # Small fee for holding positions
            
            # Combine components
            reward = (
                portfolio_return * 1.0 +  # Base return
                excess_return * 0.2 +     # Reward for beating market
                holding_cost              # Holding cost penalty
            )
            
            return float(reward)
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0