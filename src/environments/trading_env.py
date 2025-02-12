import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, lookback=10):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.lookback = lookback  # Number of historical time steps to include
        
        logger.info(f"Initialized environment with {self.max_steps + 1} data points")
        
        # Define action space (0: hold, 1-4: buy 10-40%, 5-8: sell 10-40%)
        self.action_space = spaces.Discrete(9)
        
        # Define buy and sell percentages (both 10-40%)
        self.buy_percentages = [0.1, 0.2, 0.3, 0.4]  # 10-40%
        self.sell_percentages = [0.1, 0.2, 0.3, 0.4]  # 10-40%
        
        # Define observation space (historical OHLCV + portfolio value)
        # 5 features (OHLCV) * lookback periods + 1 portfolio value
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(5 * lookback + 1,),  
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
        try:
            end_idx = self.current_step
            start_idx = max(0, end_idx - self.lookback + 1)
            
            # Get historical data
            historical_data = self.data.iloc[start_idx:end_idx + 1]
            
            # Pad with zeros if not enough historical data
            if len(historical_data) < self.lookback:
                pad_length = self.lookback - len(historical_data)
                pad_data = pd.DataFrame(0, index=range(pad_length), 
                                      columns=historical_data.columns)
                historical_data = pd.concat([pad_data, historical_data])
            
            # Extract OHLCV values
            ohlcv_data = []
            for _, row in historical_data.iterrows():
                ohlcv_data.extend([
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume'])
                ])
            
            # Calculate current portfolio value
            current_price = float(self.data.iloc[self.current_step]['close'])
            portfolio_value = self.balance + (self.position * current_price)
            
            # Combine historical OHLCV with portfolio value
            obs = np.array(ohlcv_data + [portfolio_value], dtype=np.float32)
            
            return obs
            
        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            # Return zero array with correct shape
            return np.zeros(5 * self.lookback + 1, dtype=np.float32)
    
    def _calculate_reward(self):
        if self.current_step == 0:
            return 0.0
            
        try:
            # Calculate current total value
            current_price = float(self.data.iloc[self.current_step]['close'])
            prev_price = float(self.data.iloc[self.current_step-1]['close'])
            current_portfolio_value = self.balance + (self.position * current_price)
            prev_portfolio_value = self.balance + (self.position * prev_price)
            
            # Calculate absolute value changes
            portfolio_value_change = current_portfolio_value - prev_portfolio_value
            market_value_change = (current_price - prev_price) * self.position
            
           
            
            # Combine components with appropriate scaling
            reward = (
                portfolio_value_change  # Direct value change
                    # Bonus for beating market            # Penalty for holding positions
            )
            
            # Scale reward to make it more manageable for learning
            reward = reward / 100.0  # Scale down large absolute values
            
            return float(reward)
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0